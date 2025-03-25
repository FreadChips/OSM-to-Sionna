# OSM-to-Sionna
Using Sionna to generate coverage map with openstreetmap automatically

---

## Workflow Overview

This process involves three main steps to generate and process map data:

1. **Extract OSM Data with Python**  
   Run the `1catchdata.py` script in Python. During execution, select the desired range and quantity of maps to obtain the corresponding OSM file.

2. **Batch Process in Blender**  
   In Blender's Scripting workspace, run the `2blenderscripting.py` script. This script will batch process the data and generate the corresponding XML files.
![image](https://github.com/user-attachments/assets/33981bc9-ff23-45bb-a7c5-da5b0835ab0a)

3. **Generate the Coverage Map with Python**  
   Finally, run the `3sionnatrain.py` script in Python to generate the corresponding coverage map.

---

