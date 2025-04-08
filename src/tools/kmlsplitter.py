import os
import sys
import xml.etree.ElementTree as ET
import math

def indent(elem, level=0):
    """Helper function to indent XML output for readability."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def split_kml(input_file, output_dir, num_files=None):
    # Parse the input KML file.
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Determine the namespace if present.
    ns = {}
    if root.tag.startswith("{"):
        ns_uri = root.tag[root.tag.find("{")+1:root.tag.find("}")]
        ns["kml"] = ns_uri
    else:
        ns["kml"] = ""

    # Find the Document element (assuming one exists)
    document = root.find("kml:Document", ns) if ns["kml"] else root.find("Document")
    if document is None:
        print("No Document element found in the KML file.")
        return

    # Get all Placemark elements in the Document.
    placemarks = document.findall("kml:Placemark", ns) if ns["kml"] else document.findall("Placemark")
    total_pm = len(placemarks)
    if total_pm == 0:
        print("No Placemark elements found in the KML file.")
        return

    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Use a base filename derived from the input name.
    basename = os.path.splitext(os.path.basename(input_file))[0]

    # Determine how to group placemarks.
    if num_files is None:
        # Default behavior: one Placemark per file.
        groups = [[pm] for pm in placemarks]
    else:
        try:
            num_files = int(num_files)
            if num_files <= 0:
                raise ValueError("Number of files must be a positive integer.")
        except ValueError as e:
            print("Invalid number of files:", e)
            return

        # Calculate group size: Distribute placemarks as evenly as possible.
        group_size = math.ceil(total_pm / num_files)
        groups = [placemarks[i:i+group_size] for i in range(0, total_pm, group_size)]
        # If for some reason we end up with fewer groups than requested,
        # we ensure that we have exactly num_files groups (some groups may be empty).
        while len(groups) < num_files:
            groups.append([])

    # Create a new KML file for each group.
    for idx, group in enumerate(groups, start=1):
        # Create a new KML root using the original root tag and attributes.
        new_kml = ET.Element(root.tag, root.attrib)

        # Create a new Document element. We use the local tag name (strip any namespace).
        document_tag = document.tag.split("}")[1] if "}" in document.tag else document.tag
        new_doc = ET.SubElement(new_kml, document_tag)

        # (Optional) If you have non-Placemark elements like styles, you can copy them here.
        # For simplicity, only the grouped Placemark elements are added.
        for placemark in group:
            # You can use a deep copy if necessary, but here we simply append.
            new_doc.append(placemark)

        # Pretty-print the XML.
        indent(new_kml)

        # Define output file name.
        output_file = os.path.join(output_dir, f"{basename}_{idx}.kml")
        ET.ElementTree(new_kml).write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"Created: {output_file}")

def main():
    usage = "Usage: python split_kml.py input_file.kml [output_folder] [num_files]\n" \
            " - input_file.kml: The path to the input KML file.\n" \
            " - output_folder: (Optional) Folder to save the output files. Defaults to 'output_kml_files'.\n" \
            " - num_files: (Optional) The number of output files to split the placemarks into. " \
            "If not provided, each Placemark is written to its own file."
    
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_kml_files"
    num_files = sys.argv[3] if len(sys.argv) > 3 else None

    split_kml(input_file, output_dir, num_files)

if __name__ == "__main__":
    main()
