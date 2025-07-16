import pandas as pd
from pathlib import Path

# Define paths
root_dir = Path("E:/GRR")
output_dir = root_dir / "Combined_CSVs"
output_dir.mkdir(exist_ok=True)

csv_data = {}

for top_folder in root_dir.iterdir():
    if top_folder.is_dir() and top_folder.name != "Combined_CSVs":
        # Extract experiment suffix (all1, all2, all3)
        exp_suffix = top_folder.name.split('_')[-1]
        
        for second_level in top_folder.iterdir():
            if second_level.is_dir():
                # Extract batch size from second level folder name
                if '_bs_' in second_level.name:
                    batch_size = second_level.name.split('_bs_')[-1]
                else:
                    # Fallback if naming convention differs
                    parts = second_level.name.split('_')
                    batch_size = parts[-1] if parts[-1].isdigit() else "unknown"
                
                for third_level in second_level.iterdir():
                    if third_level.is_dir():
                        # Extract adaptive method (remove 'results_' prefix)
                        adaptive_method = third_level.name.replace("results_", "")
                        
                        # Process all CSV files
                        for csv_file in third_level.glob('*.csv'):
                            df = pd.read_csv(csv_file)
                            
                            # Add metadata columns
                            df['experiment_suffix'] = exp_suffix
                            df['batch_size'] = batch_size
                            df['adaptive_method'] = adaptive_method
                            
                            # Store in dictionary
                            csv_name = csv_file.name
                            if csv_name not in csv_data:
                                csv_data[csv_name] = []
                            csv_data[csv_name].append(df)

# Combine and save all CSV files
for csv_name, dfs in csv_data.items():
    combined_df = pd.concat(dfs, ignore_index=True)
    output_path = output_dir / csv_name
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined file: {output_path}")

print("Processing completed successfully!")
print(f"Combined files saved to: {output_dir}")