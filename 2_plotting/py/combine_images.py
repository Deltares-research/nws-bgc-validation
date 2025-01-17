# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:40:19 2024

@author: lorinc
"""

from PIL import Image
import os

def combine_images(image_paths, output_path, layout="horizontal", grid_size=(2, 2), dpi=300, tight_layout=True):
    """
    Combines PNG images into a single image.

    Parameters:
        image_paths (list of str): List of paths to the PNG images to combine.
        output_path (str): Path to save the combined image.
        layout (str): Layout type ('horizontal', 'vertical', or 'grid').
        grid_size (tuple): Size of the grid for 'grid' layout (rows, columns).

    Returns:
        None
    """
    # Open all images
    images = [Image.open(img).convert("RGBA") for img in image_paths]

    # Determine the layout
    if layout == "horizontal":
        # Combine horizontally
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        combined_image = Image.new("RGBA", (total_width, max_height))

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

    elif layout == "vertical":
        # Combine vertically
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        combined_image = Image.new("RGBA", (max_width, total_height))

        y_offset = 0
        for img in images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.height

    elif layout == "grid":
        # Combine into a grid
        rows, cols = grid_size
        if len(images) > rows * cols:
            raise ValueError("Grid size is too small for the number of images.")
        
        # Calculate the total width and height based on the actual size of each image
        total_width = sum(img.width for img in images[:cols])
        total_height = sum(img.height for img in images[::cols])
    
        combined_image = Image.new("RGBA", (total_width, total_height))
    
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x_offset = col * img.width  # use img.width for each image
            y_offset = row * img.height  # use img.height for each image
            combined_image.paste(img, (x_offset, y_offset))
    else:
        raise ValueError("Invalid layout. Choose 'horizontal', 'vertical', or 'grid'.")
        
    # Apply tight layout by cropping extra transparent areas if enabled
    if tight_layout:
        bbox = combined_image.getbbox()
        if bbox:
            combined_image = combined_image.crop(bbox)

    # Save the combined image
    combined_image.save(output_path, "PNG", dpi=(dpi, dpi))
    print(f"Combined image saved to {output_path}")
#%%
# Example usage
start_year=2015
end_year=2017
fig_type = "transects"
variables = ['NO3']#,"NO3","PO4","CHL", "PH", "OXY"] #"PCO2", 
slice2d = "transect"
keyword1 = "NORWAY2" #"non-fixed", "fixed"
keyword2 = "depth"
layout = "grid" # "horizontal", "vertical" # Plot layout

parent_dir = fr"p:\11209810-cmems-nws\figures\{fig_type}\{start_year}_{end_year}"
    
for variable in variables:
    print(f"Running {variable}")
    #combine keywords
    keywords = [keyword2, variable, slice2d, keyword1]
    
    # Output path for the combined image
    output_file = os.path.join(parent_dir, f'combined_{start_year}_{end_year}_{variable}_{slice2d}_{keyword1}.png')
    
    #Remove existing, if present
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Removing.")
        os.remove(output_file)
    
    # Specify the paths to PNG images
    image_files = []
        
    # Walk through the directory
    for root, _, files in os.walk(parent_dir):
        for file in files:
            # Check if any keyword is in the filename
            if all(keyword in file for keyword in keywords):
                image_files.append(os.path.join(root, file))
    
    #Filter out figures of observation count
    image_files = [path for path in image_files if "count" not in path]
    
    if len(image_files) > 4:
        grid_size = (3, 3) #(rows, columns)
    else:
        grid_size = (2, 2) #(rows, columns)
    
    # Combine images into a grid (e.g., 2 rows, 2 columns)
    combine_images(image_files, output_file, layout=layout, grid_size=grid_size, dpi=300, tight_layout=True)
