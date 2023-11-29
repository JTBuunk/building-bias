import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from collections import defaultdict
from collections import defaultdict
import pandas as pd

# Load GeoDataFrames
# Replace with own datasets

reference_gdf = gpd.read_file("F:\\Internship\\Data\\building_data\\ref_buildings_try1.shp")
osm_gdf = gpd.read_file("F:\\Internship\\Data\\building_data\\OSM_selected_new.shp")
omf_gdf = gpd.read_file("F:\\Internship\\Data\\building_data\\OMF_selected_new.shp")
bing_gdf = gpd.read_file("F:\\Internship\\Data\\building_data\\bing_selected_new.shp")
google_gdf = gpd.read_file("F:\\Internship\\Data\\building_data\\google_selected_new.shp")
study_areas_gdf = gpd.read_file("F:\\Internship\\Data\\SA_final3.shp")


print(f"Total buildings in reference dataset (before clipping): {len(reference_gdf)}")
print(f"Total buildings in OSM dataset (before clipping): {len(osm_gdf)}")
print(f"Total buildings in OMF dataset (before clipping): {len(omf_gdf)}")
print(f"Total buildings in Bing dataset (before clipping): {len(bing_gdf)}")
print(f"Total buildings in Google dataset (before clipping): {len(google_gdf)}")


# Initialize counts for each dataset
datasets = {
    "Google": google_gdf,
    "OSM": osm_gdf,
    "Bing": bing_gdf,
    "OMF": omf_gdf,

}

# Create index for files due to FID's not read by python
reference_gdf['ref_building_id'] = reference_gdf.index
osm_gdf['ai_building_id'] = osm_gdf.index
bing_gdf['ai_building_id'] = bing_gdf.index
google_gdf['ai_building_id'] = google_gdf.index
omf_gdf['ai_building_id'] = omf_gdf.index
study_areas_gdf['tile_id'] = study_areas_gdf.index

# Adjust if necessary
iou_threshold = 0.5

# Functions for calculating IoU
def compute_iou_max(a, b):
    intersection = a.intersection(b).area
    union = a.area + b.area - intersection
    return intersection / union if union != 0 else 0

def compute_combined_iou(ai_building_geometry, ref_buildings_geometries):
    # Calculate the total intersection area
    total_intersection_area = sum(ai_building_geometry.intersection(ref_geometry).area for ref_geometry in ref_buildings_geometries)

    # Calculate the total area of the union
    total_ai_building_area = ai_building_geometry.area
    total_ref_buildings_area = sum(ref_geometry.area for ref_geometry in ref_buildings_geometries)
    total_union_area = total_ai_building_area + total_ref_buildings_area - total_intersection_area

    # Compute the IoU
    combined_iou = total_intersection_area / total_union_area if total_union_area != 0 else 0
    return combined_iou


# Store results
results = defaultdict(list)

# Dictionary to track area and count information for each study area
study_area_stats = defaultdict(lambda: defaultdict(dict))

# Initialize a set to track reference buildings that have been matched
matched_reference_buildings = set()



# Loop through each study area
for _, study_area in study_areas_gdf.iterrows():
    study_area_id = study_area['tile_id']


    # Clip the reference dataset according to the geometry of the current study area
    clipped_reference_gdf = reference_gdf[reference_gdf.geometry.intersects(study_area.geometry)]

    # Get area and count of reference dataset for each tile
    ref_area_total = clipped_reference_gdf.geometry.area.sum()
    ref_building_count = len(clipped_reference_gdf)

    # Store the reference area and count in the stats dictionary
    study_area_stats[study_area_id]['ref'] = {
        'area_total': ref_area_total,
        'building_count': ref_building_count
    }


    # Loop through each building dataset
    for dataset_name, dataset_gdf in datasets.items():

        # Clip buildings of the current dataset to the study area
        clipped_dataset_gdf = dataset_gdf[dataset_gdf.geometry.intersects(study_area.geometry)]

        # Get area and count of AI dataset for each tile
        ai_area_total = clipped_dataset_gdf.geometry.area.sum()
        ai_building_count = len(clipped_dataset_gdf)

        # Calculate the absolute and relative differences
        abs_diff_area = abs(ref_area_total - ai_area_total)
        rel_diff_area = abs_diff_area / ref_area_total if ref_area_total != 0 else 0
        abs_diff_count = abs(ref_building_count - ai_building_count)
        rel_diff_count = abs_diff_count / ref_building_count if ref_building_count != 0 else 0
        # diff_count_nor = (ai_building_count - ref_building_count) / ref_building_count if ref_building_count != 0 else 0

        # Store the building area and count in the stats dictionary
        study_area_stats[study_area_id][dataset_name] = {
            'area_abs': abs_diff_area,
            'area_rel': rel_diff_area,
            'count_abs': abs_diff_count,
            'count_rel': rel_diff_count,
            # 'count_nor': diff_count_nor,
        }


        # Loop through each building in the clipped dataset
        for _, dataset_building in clipped_dataset_gdf.iterrows():
            building_id = dataset_building['ai_building_id']

            # Check if the building intersects with any reference building
            overlapping_ref_buildings = clipped_reference_gdf[clipped_reference_gdf.geometry.intersects(dataset_building.geometry)]

            # If there's any intersection
            if not overlapping_ref_buildings.empty:
                # Calculate the IoU_max for each intersecting reference building and find the building with max IoU
                iou_values = [(ref_building['ref_building_id'], compute_iou_max(dataset_building.geometry, ref_building.geometry)) for _, ref_building in overlapping_ref_buildings.iterrows()]
                ref_building_id, max_iou = max(iou_values, key=lambda x: x[1])


                # Get the geometries of all overlapping reference buildings
                ref_geometries = overlapping_ref_buildings.geometry.tolist()

                # Calculate the combined IoU for the dataset building and all intersecting reference buildings
                combined_iou = compute_combined_iou(dataset_building.geometry, ref_geometries)

                # Check for true positive or false negative for max iou
                if max_iou >= iou_threshold:
                    tp = 1
                    fn = 0
                else:
                    tp = 0
                    fn = 1

                # Check for true positive or false negative for combined IoU
                if combined_iou >= iou_threshold:
                    tp_com = 1
                    fn_com = 0
                else:
                    tp_com = 0
                    fn_com = 1

                # Add the matched reference building ID to the set
                matched_reference_buildings.add(ref_building_id)

                # Append results
                results[dataset_name].append({
                    'ai_building_id': building_id,
                    'ref_building_id': ref_building_id,
                    'study_area_id': study_area_id,
                    'poverty': study_area['poverty'], # Replace with names of sensitive variables in tile dataset
                    'urban': study_area['catchment'],
                    'ref_building_area': overlapping_ref_buildings.loc[ref_building_id].geometry.area,
                    'vul_pop': study_area['Vul_pop_go'],
                    'pop_dens': study_area['dens_ad3'],
                    'elev': study_area['elev'],
                    'RWI': study_area['rwi'],
                    'popdens_fb': study_area['popdens_fb'],
                    'max_iou': max_iou,
                    'combined_iou': combined_iou,
                    'tp': tp,
                    'fp': 0,  # FP is 0 here because there is an intersection
                    'fn': fn,
                    'tp_com': tp_com,
                    'fp_com': 0,
                    'fn_com': fn_com
                })
            else:
                # If there is no intersection, it's a false positive
                results[dataset_name].append({
                    'ai_building_id': building_id,
                    'ref_building_id': None,  # No corresponding reference building
                    'study_area_id': study_area_id,
                    'poverty': study_area['poverty'], # Replace with names of sensitive variables in tile dataset
                    'urban': study_area['catchment'],
                    'ref_building_area': None,
                    'vul_pop': study_area['Vul_pop_go'],
                    'pop_dens': study_area['dens_ad3'],
                    'elev': study_area['elev'],
                    'RWI': study_area['rwi'],
                    'popdens_fb': study_area['popdens_fb'],
                    'max_iou': 0,  # No intersection, so IoU is 0
                    'combined_iou': 0,
                    'ref_building_area': None,
                    'tp': 0,
                    'fp': 1,
                    'fn': 0,  # FN is not applicable here as there's no reference building
                    'tp_com': 0,
                    'fp_com': 1,
                    'fn_com': 0
                })

        # Check for false negatives in the reference buildings that were not matched
        for ref_building_id, ref_building in clipped_reference_gdf.iterrows():
            if ref_building_id not in matched_reference_buildings:
                # This reference building was not matched with any AI building, so it's a FN
                results[dataset_name].append({
                    'ai_building_id': None,  # No corresponding AI building
                    'ref_building_id': ref_building_id,
                    'study_area_id': study_area_id,
                    'poverty': study_area['poverty'], # Replace with names of sensitive variables in tile dataset
                    'urban': study_area['catchment'],
                    'vul_pop': study_area['Vul_pop_go'],
                    'pop_dens': study_area['dens_ad3'],
                    'elev': study_area['elev'],
                    'RWI': study_area['rwi'],
                    'popdens_fb': study_area['popdens_fb'],
                    'max_iou': 0,  # No intersection, so IoU is 0
                    'combined_iou': 0,
                    'ref_building_area': ref_building.geometry.area,
                    'tp': 0,
                    'fp': 0,  # FP is not applicable here
                    'fn': 1,
                    'tp_com': 0,
                    'fp_com': 0,
                    'fn_com': 1
                })



# Include tile statistics in results
for dataset_name, dataset_results in results.items():
    for result in dataset_results:
        study_area_id = result['study_area_id']
        # Append the area and building count differences to each result
        result.update({
            'area_abs': study_area_stats[study_area_id][dataset_name]['area_abs'],
            'area_rel': study_area_stats[study_area_id][dataset_name]['area_rel'],
            'count_abs': study_area_stats[study_area_id][dataset_name]['count_abs'],
            'count_rel': study_area_stats[study_area_id][dataset_name]['count_rel'],
            # 'count_nor': study_area_stats[study_area_id][dataset_name]['count_nor'],
        })


for dataset_name, dataset_results in results.items():
    df = pd.DataFrame(dataset_results)
    df.to_excel(f"phl_{dataset_name}_results_try4.xlsx", index=False)

# # Print total clipped counts for each dataset
# for dataset_name, count in clipped_counts.items():
#     print(f"Total clipped buildings in {dataset_name} dataset: {count}")



# Next section calculates and saves statistics on tile level

# Initialize a dictionary to store the aggregated results for each study area with unique keys
study_area_summary = defaultdict(lambda: defaultdict(int))
# Dictionary to accumulate total building area and count per study area with unique keys
building_stats = defaultdict(lambda: {'total_area': 0, 'count': 0})

# Populate the summary dictionary with aggregated results
for dataset_name, dataset_results in results.items():
    for result in dataset_results:
        # Create a unique key for the study_area_id within this dataset
        unique_key = (dataset_name, result['study_area_id'])

        if result['ref_building_area'] is not None:
            building_stats[unique_key]['total_area'] += result['ref_building_area']
            building_stats[unique_key]['count'] += 1

        # Add the sensitive variable values just once per study area
        if 'poverty' not in study_area_summary[unique_key]:
            study_area_summary[unique_key]['poverty'] = result['poverty']
            study_area_summary[unique_key]['pop_dens'] = result['pop_dens']
            study_area_summary[unique_key]['vul_pop'] = result['vul_pop']
            study_area_summary[unique_key]['elev'] = result['elev']
            study_area_summary[unique_key]['RWI'] = result['RWI']
            study_area_summary[unique_key]['urban'] = result['urban']
            study_area_summary[unique_key]['popdens_fb'] = result['popdens_fb']
        # Sum values
        study_area_summary[unique_key]['tp'] += result['tp']
        study_area_summary[unique_key]['fp'] += result['fp']
        study_area_summary[unique_key]['fn'] += result['fn']

        # Add the area and count stats
        study_area_summary[unique_key]['area_abs'] = study_area_stats[result['study_area_id']][dataset_name]['area_abs']
        study_area_summary[unique_key]['area_rel'] = study_area_stats[result['study_area_id']][dataset_name]['area_rel']
        study_area_summary[unique_key]['count_abs'] = study_area_stats[result['study_area_id']][dataset_name]['count_abs']
        study_area_summary[unique_key]['count_rel'] = study_area_stats[result['study_area_id']][dataset_name]['count_rel']
        # study_area_summary[unique_key]['count_nor'] = study_area_stats[result['study_area_id']][dataset_name]['count_nor']


# Calculate the average building size for the reference dataset within each study area
for unique_key in study_area_summary:
    total_area = building_stats[unique_key]['total_area']
    count = building_stats[unique_key]['count']
    study_area_summary[unique_key]['avg_building_size'] = total_area / count if count > 0 else None
    study_area_summary[unique_key]['building_dens'] = count


# Convert the summary dictionary to a DataFrame and export to Excel
study_area_summary_df = pd.DataFrame.from_dict(study_area_summary, orient='index').reset_index()

# Rename the columns to reflect the unique keys
study_area_summary_df.rename(columns={'level_0': 'dataset_name', 'level_1': 'study_area_id'}, inplace=True)

#
# Export each dataset's summary to a separate Excel file
for dataset_name in datasets.keys():
    # Filter the study_area_summary_df DataFrame for the current dataset_name
    filtered_df = study_area_summary_df[study_area_summary_df['dataset_name'] == dataset_name]
    # Drop the dataset_name column for the excel file
    filtered_df = filtered_df.drop(columns=['dataset_name'])
    # Define the Excel file name based on the dataset name
    excel_file_name = f'phl_{dataset_name}_tile_results_try4.xlsx'
    # Write the filtered DataFrame to the Excel file
    filtered_df.to_excel(excel_file_name, index=False)
