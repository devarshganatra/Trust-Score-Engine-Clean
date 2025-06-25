#!/usr/bin/env python3
"""
Data Converter Script
Converts Python dictionary format to proper JSON format
"""

import ast
import json
import os
from tqdm import tqdm

def convert_file(input_file, output_file):
    """Convert Python dict format to JSON format"""
    print(f"Converting {input_file} to {output_file}...")
    
    converted_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc="Converting"):
                line = line.strip()
                if line:
                    try:
                        # Parse Python dict string
                        data = ast.literal_eval(line)
                        # Write as JSON
                        json.dump(data, f_out, ensure_ascii=False)
                        f_out.write('\n')
                        converted_count += 1
                    except Exception as e:
                        print(f"Error converting line: {e}")
                        continue
    
    print(f"✅ Converted {converted_count} records")
    return converted_count

def main():
    """Main conversion function"""
    print("🔄 Data Format Converter")
    print("=" * 30)
    
    # Create backup directory
    os.makedirs('data/backup', exist_ok=True)
    
    # Convert reviews file
    reviews_input = 'data/reviews_Electronics_5.json'
    reviews_output = 'data/reviews_Electronics_5_converted.json'
    
    if os.path.exists(reviews_input):
        print(f"Backing up original file...")
        os.system(f"cp {reviews_input} data/backup/reviews_Electronics_5_original.json")
        
        reviews_count = convert_file(reviews_input, reviews_output)
        print(f"Reviews: {reviews_count} records converted")
    else:
        print(f"❌ Reviews file not found: {reviews_input}")
    
    # Convert products file
    products_input = 'data/meta_Electronics.json'
    products_output = 'data/meta_Electronics_converted.json'
    
    if os.path.exists(products_input):
        print(f"Backing up original file...")
        os.system(f"cp {products_input} data/backup/meta_Electronics_original.json")
        
        products_count = convert_file(products_input, products_output)
        print(f"Products: {products_count} records converted")
    else:
        print(f"❌ Products file not found: {products_input}")
    
    print("\n✅ Conversion completed!")
    print("\n📋 Next steps:")
    print("1. Use the converted files:")
    print(f"   - Reviews: {reviews_output}")
    print(f"   - Products: {products_output}")
    print("2. Run the pipeline:")
    print(f"   python3 main.py run --reviews {reviews_output} --products {products_output}")

if __name__ == "__main__":
    main() 