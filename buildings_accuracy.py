import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from collections import defaultdict
from collections import defaultdict
import pandas as pd

# load GeoDataFrames and replace with own datasets
reference_gdf = gpd.read_file("path_to_file")
osm_gdf = gpd.read_file("path_to_file")
bing_gdf = gpd.read_file("path_to_file")
google_gdf = gpd.read_file("path_to_file")
omf_gdf = gpd.read_file("path_to_file")
google_conf_gdf = gpd.read_file("path_to_file")
study_areas_gdf = gpd.read_file("path_to_file")

# initialize counts for each dataset
datasets = {
    "OMF": omf_gdf,
    "OSM": osm_gdf,
    "Bing": bing_gdf,
    "Google": google_gdf,
    "Google_conf": google_conf_gdf
}

# create index for files due to FID's not read by python
reference_gdf['ref_building_id'] = reference_gdf.index
osm_gdf['ai_building_id'] = osm_gdf.index
bing_gdf['ai_building_id'] = bing_gdf.index
google_gdf['ai_building_id'] = google_gdf.index
omf_gdf['ai_building_id'] = omf_gdf.index
study_areas_gdf['tile_id'] = study_areas_gdf.index
google_conf_gdf['ai_building_id'] = google_conf_gdf.index

# adjust if necessary
iou_threshold = 0.5

# functions for calculating IoU
def compute_iou_max(a, b):
    intersection = a.intersection(b).area
    union = a.area + b.area - intersection
    return intersection / union if union != 0 else 0

def compute_combined_iou(ai_building_geometry, ref_buildings_geometries):
    # calculate the total intersection area
    total_intersection_area = sum(ai_building_geometry.intersection(ref_geometry).area for ref_geometry in ref_buildings_geometries)

    # calculate the total area of the union
    total_ai_building_area = ai_building_geometry.area
    total_ref_buildings_area = sum(ref_geometry.area for ref_geometry in ref_buildings_geometries)
    total_union_area = total_ai_building_area + total_ref_buildings_area - total_intersection_area

    # compute the IoU
    combined_iou = total_intersection_area / total_union_area if total_union_area != 0 else 0
    return combined_iou


# store results
results = defaultdict(list)

# dictionary to track area and count information for each study area
study_area_stats = defaultdict(lambda: defaultdict(dict))


# loop through each study area
for _, study_area in study_areas_gdf.iterrows():
    study_area_id = study_area['tile_id']


    # clip the reference dataset according to the geometry of the current study area
    clipped_reference_gdf = reference_gdf[reference_gdf.geometry.intersects(study_area.geometry)]

    # get area and count of reference dataset for each tile
    ref_area_total = clipped_reference_gdf.geometry.area.sum()
    ref_building_count = len(clipped_reference_gdf)
    ref_building_size = ref_area_total / ref_building_count


    # loop through each building dataset
    for dataset_name, dataset_gdf in datasets.items():

        # clip buildings of the current dataset to the study area
        clipped_dataset_gdf = dataset_gdf[dataset_gdf.geometry.intersects(study_area.geometry)]

        # initialize a set to track reference buildings that have been matched
        matched_reference_buildings = set()

        # get area and count of AI dataset for each tile
        ai_area_total = clipped_dataset_gdf.geometry.area.sum()
        ai_building_count = len(clipped_dataset_gdf)

        # calculate nbc
        nbc = (ai_building_count - ref_building_count) / ref_building_count if ref_building_count != 0 else 0

        # store nbc and building size in the stats dictionary
        study_area_stats[study_area_id][dataset_name] = {
            'nbc': nbc,
            'building_size': ref_building_size,
        }


        # loop through each building in the clipped dataset
        for _, dataset_building in clipped_dataset_gdf.iterrows():
            building_id = dataset_building['ai_building_id']

            # check if the building intersects with any reference building
            overlapping_ref_buildings = clipped_reference_gdf[clipped_reference_gdf.geometry.intersects(dataset_building.geometry)]

            # if there's any intersection
            if not overlapping_ref_buildings.empty:
                # calculate the IoU_max for each intersecting reference building and find the building with max IoU
                iou_values = [(ref_building['ref_building_id'], compute_iou_max(dataset_building.geometry, ref_building.geometry)) for _, ref_building in overlapping_ref_buildings.iterrows()]
                ref_building_id, max_iou = max(iou_values, key=lambda x: x[1])


                # get the geometries of all overlapping reference buildings
                ref_geometries = overlapping_ref_buildings.geometry.tolist()

                # calculate the combined IoU for the dataset building and all intersecting reference buildings
                combined_iou = compute_combined_iou(dataset_building.geometry, ref_geometries)

                # check for true positive or false negative for max iou
                if max_iou >= iou_threshold:
                    tp = 1
                    fn = 0
                else:
                    tp = 0
                    fn = 1

                # check for true positive or false negative for combined IoU
                if combined_iou >= iou_threshold:
                    tp_com = 1
                    fn_com = 0
                else:
                    tp_com = 0
                    fn_com = 1

                # add the matched reference building ID to the set
                matched_reference_buildings.add(ref_building_id)

                # append results
                results[dataset_name].append({
                    'ai_building_id': building_id,
                    'ref_building_id': ref_building_id,
                    'study_area_id': study_area_id,
                    'max_iou': max_iou,
                    'combined_iou': combined_iou,
                    'tp': tp,
                    'fp': 0,  # FP is 0 here because there is an intersection
                    'fn': fn,
                    'tp_com': tp_com,
                    'fp_com': 0,
                    'fn_com': fn_com,
                    'ref_building_area': overlapping_ref_buildings.loc[ref_building_id].geometry.area,

                    # replace with names of sensitive variables in tile dataset
                    'poverty': study_area['name_of_column'],
                    'urban': study_area['name_of_column'],
                    'RWI': study_area['name_of_column'],
                    'pop_dens': study_area['name_of_column'],
                })
            else:
                # if there is no intersection, it's a false positive
                results[dataset_name].append({
                    'ai_building_id': building_id,
                    'ref_building_id': None,  # no corresponding reference building
                    'study_area_id': study_area_id,
                    'max_iou': 0,  # no intersection, so IoU is 0
                    'combined_iou': 0,
                    'tp': 0,
                    'fp': 1,
                    'fn': 0,  # FN is not applicable here as there's no reference building
                    'tp_com': 0,
                    'fp_com': 1,
                    'fn_com': 0,
                    'ref_building_area': None,

                    # replace with names of sensitive variables in tile dataset
                    'poverty': study_area['name_of_column'],
                    'urban': study_area['name_of_column'],
                    'RWI': study_area['name_of_column'],
                    'pop_dens': study_area['name_of_column'],
                })

        # check for false negatives in the reference buildings that were not matched
        for ref_building_id, ref_building in clipped_reference_gdf.iterrows():
            if ref_building_id not in matched_reference_buildings:
                # this reference building was not matched with any AI building, so it's a FN
                results[dataset_name].append({
                    'ai_building_id': None,  # no corresponding AI building
                    'ref_building_id': ref_building_id,
                    'study_area_id': study_area_id,
                    'max_iou': 0,  # no intersection, so IoU is 0
                    'combined_iou': 0,
                    'tp': 0,
                    'fp': 0,  # FP is not applicable here
                    'fn': 1,
                    'tp_com': 0,
                    'fp_com': 0,
                    'fn_com': 1,
                    'ref_building_area': ref_building.geometry.area,

                    # replace with names of sensitive variables in tile dataset
                    'poverty': study_area['name_of_column'],
                    'urban': study_area['name_of_column'],
                    'RWI': study_area['name_of_column'],
                    'pop_dens': study_area['name_of_column'],
                })



# include tile statistics in results
for dataset_name, dataset_results in results.items():
    for result in dataset_results:
        study_area_id = result['study_area_id']

        # append nbc to each result
        result.update({
            'nbc': study_area_stats[study_area_id][dataset_name]['nbc'],
        })


for dataset_name, dataset_results in results.items():
    df = pd.DataFrame(dataset_results)
    df.to_excel(f"{dataset_name}_results.xlsx", index=False)



# next section calculates and saves statistics on tile level

# initialize a dictionary to store the aggregated results for each study area with unique keys
study_area_summary = defaultdict(lambda: defaultdict(int))

# dictionary to accumulate total building area and count per study area with unique keys
building_stats = defaultdict(lambda: {'total_area': 0, 'count': 0})

# populate the summary dictionary with aggregated results
for dataset_name, dataset_results in results.items():
    for result in dataset_results:
        # create a unique key for the study_area_id within this dataset
        unique_key = (dataset_name, result['study_area_id'])

        if result['ref_building_area'] is not None:
            building_stats[unique_key]['total_area'] += result['ref_building_area']
            building_stats[unique_key]['count'] += 1

        # ensure the names of the result section here are identical to previous section
        # add the sensitive variable values just once per study area
        if 'name_of_first_sensitive_variable' not in study_area_summary[unique_key]:
            study_area_summary[unique_key]['poverty'] = result['name_of_column']
            study_area_summary[unique_key]['pop_dens'] = result['name_of_column']
            study_area_summary[unique_key]['RWI'] = result['name_of_column']
            study_area_summary[unique_key]['urban'] = result['name_of_column']


        # sum values
        study_area_summary[unique_key]['tp'] += result['tp']
        study_area_summary[unique_key]['fp'] += result['fp']
        study_area_summary[unique_key]['fn'] += result['fn']

        # add nbc stats
        study_area_summary[unique_key]['nbc'] = study_area_stats[result['study_area_id']][dataset_name]['nbc']



# calculate the average building size for the reference dataset within each study area
for unique_key in study_area_summary:
    total_area = building_stats[unique_key]['total_area']
    count = building_stats[unique_key]['count']
    study_area_summary[unique_key]['building_size'] = total_area / count if count > 0 else None
    study_area_summary[unique_key]['building_dens'] = count


# convert the summary dictionary to a DataFrame and export to Excel
study_area_summary_df = pd.DataFrame.from_dict(study_area_summary, orient='index').reset_index()

# rename the columns to reflect the unique keys
study_area_summary_df.rename(columns={'level_0': 'dataset_name', 'level_1': 'study_area_id'}, inplace=True)


# export each dataset's summary to a separate Excel file
for dataset_name in datasets.keys():
    filtered_df = study_area_summary_df[study_area_summary_df['dataset_name'] == dataset_name]
    filtered_df = filtered_df.drop(columns=['dataset_name'])
    excel_file_name = f'{dataset_name}_tile_results.xlsx'
    filtered_df.to_excel(excel_file_name, index=False)

