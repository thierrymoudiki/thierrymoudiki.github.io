# modification of config created here: https://gist.github.com/cscorley/9144544
# Useful nbconvert documentation:
#   https://nbconvert.readthedocs.io/en/latest/config_options.html#cli-flags-and-aliases
import os
import re
import sys
from jupyter_core.paths import jupyter_path
import pathlib
import argparse

'''
Expects to be called as: `jupyter nbconvert PATH_TO_NOTEBOOK.ipynb --flag --flag ...
'''

'''
Parse CLI arguments
'''
parser = argparse.ArgumentParser()
parser.add_argument('ipynb_path', type=str)
args, __ = parser.parse_known_args()

'''
Input paths
'''
this_script_dir: str = os.path.abspath(pathlib.Path(__file__).parent.resolve())
template_path: str = os.path.abspath(os.path.join(this_script_dir, 'nbconvert_jekyll.tpl'))
ipynb_file_name: str = os.path.basename(args.ipynb_path)
'''
Output paths
'''
output_abs_dir: str = os.path.abspath(pathlib.Path(args.ipynb_path).parent.resolve()) # '/Users/Desktop/_posts/YYYY-MM-DD-post-name/'
output_relative_dir: str = '/'.join(args.ipynb_path.split('/')[:-1]) # '_posts/YYYY-MM-DD-post-name/'
output_image_abs_dir: str = os.path.abspath(os.path.join(output_abs_dir, 'markdown_images/')) # '/Users/Desktop/_posts/YYYY-MM-DD-post-name/markdown_images/'
output_image_relative_dir: str = os.path.join(output_relative_dir, 'markdown_images/') # '_posts/YYYY-MM-DD-post-name/markdown_images/'
# Make sure our `base_file_name` doesn't start with "YYYY-MM-DD", otherwise Jekyll will be tripped up when creating image/pdf/support files b/c 
# it only expects the 'YYYY-MM-DD' prefix for .md posts
# NOTE: This means that we need to add the 'YYYY-MM-DD' prefix to our .md file after creating it
base_file_name_with_date_prefix: str = ipynb_file_name.lower().replace(' ', '-').replace('.ipynb', '') # 'YYYY-MM-DD-post-name'
base_file_name: str = re.sub(r'^\d{4}\-\d{2}\-\d{2}\-', '', base_file_name_with_date_prefix) # 'YYYY-MM-DD-post-name' => 'post-name'

'''
Sanity checks
'''
# assert os.path.exists(template_path), f"[ ] Couldn't find .tpl template @ {template_path}"
# print(f"[X] Successfully found .tpl template @ {template_path}")
assert os.path.exists(output_abs_dir), f"[ ] Couldn't find output directory to store .md file @ {output_abs_dir}/"
print(f"[X] Successfully found directory to save .md file @ {output_abs_dir}/")
os.makedirs(output_image_abs_dir, exist_ok=True)
assert os.path.exists(output_image_abs_dir), f"[ ] Couldn't find output directory to store images @ {output_image_abs_dir}/"
print(f"[X] Successfully found directory to store images @ {output_image_abs_dir}/")

'''
Setup configs
'''
c = get_config()
c.NbConvertApp.export_format = 'markdown'
# See: https://github.com/mpacer/nbconvert/blob/bd20c4f6959d277a9e84cf8f48456e57268aeac4/nbconvert/preprocessors/extractoutput.py
c.NbConvertApp.output_base = base_file_name # Base name for images/files/pdfs/markdown output by `nbconvert`, i.e. NAME of "NAME.md" or "NAME.png"
c.NbConvertApp.output_files_dir = output_image_abs_dir # Directory to write image outputs to
c.FilesWriter.build_directory = output_abs_dir # Directory to write .md outputs to

c.MarkdownExporter.template_path = jupyter_path('nbconvert','templates') # List of paths to user's Jupyter templates directories
c.MarkdownExporter.template_file = template_path # Path to our custom .tpl file
'''
Map image paths -> relative URLs
    i.e. "/Users/Desktop/_posts/YYYY-MM-DD-post-name/markdown_images/image.png" => "markdown_images/image.png"
'''
def path2support(path: str):
    image_file_name: str = os.path.basename(path)
    image_path: str = "markdown_images/" + image_file_name
    return image_path
c.MarkdownExporter.filters = {'path2support': path2support}