import os
import re
import subprocess
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('ipynb_path', type=str)
args = parser.parse_args()


'''
Input paths
'''
this_script_dir: str = os.path.abspath(pathlib.Path(__file__).parent.resolve())
ipynb_file_name: str = os.path.basename(args.ipynb_path)
config_script_path: str = os.path.join(this_script_dir, 'nbconvert_config.py')
'''
Output paths
'''
output_abs_dir: str = os.path.abspath(pathlib.Path(args.ipynb_path).parent.resolve()) # '/Users/Desktop/_posts/YYYY-MM-DD-post-name/'
output_relative_dir: str = '/'.join(args.ipynb_path.split('/')[:-1]) # '_posts/YYYY-MM-DD-post-name/'
output_image_abs_dir: str = os.path.abspath(os.path.join(output_abs_dir, 'markdown_images/')) # '/Users/Desktop/_posts/YYYY-MM-DD-post-name/markdown_images/'
output_image_relative_dir: str = os.path.join(output_relative_dir, 'markdown_images/') # '_posts/YYYY-MM-DD-post-name/markdown_images/'
base_file_name_with_date_prefix: str = ipynb_file_name.lower().replace(' ', '-').replace('.ipynb', '') # 'YYYY-MM-DD-post-name'
base_file_name: str = re.sub(r'^\d{4}\-\d{2}\-\d{2}\-', '', base_file_name_with_date_prefix) # 'YYYY-MM-DD-post-name' => 'post-name'
output_markdown_abs_path: str = os.path.join(output_abs_dir, base_file_name + '.md') # '/Users/Desktop/_posts/YYYY-MM-DD-post-name/post-name.md'
jekyll_markdown_abs_path: str = os.path.join(output_abs_dir, base_file_name_with_date_prefix + '.md')  # '/Users/Desktop/_posts/YYYY-MM-DD-post-name/YYYY-MM-DD-post-name.md'

print(f"Converting {ipynb_file_name} => {os.path.basename(jekyll_markdown_abs_path)}")
subprocess.run(["jupyter", "nbconvert", args.ipynb_path, "--to", "markdown", "--config", config_script_path])

# Clean up markdown
with open(output_markdown_abs_path, 'r') as fd:
    md = fd.read()
md_clean = md

# HTML cleanup
#   Remove <style> tags
md_clean = re.sub(r'\<style scoped\>(.|\n)*\<\/style\>','', md_clean, flags=re.IGNORECASE)
#   Remove <axessubplot> tags
md_clean = re.sub(r'\<\/?axessubplot:.*\n','', md_clean, flags=re.IGNORECASE)
#   Avoid "Tag '{%' was not properly terminated with regexp" errors
idxs = [ x.start() for x in re.finditer('{%', md_clean) ] + \
        [ x.start() for x in re.finditer('%}', md_clean) ] 
        # [ x.start() for x in re.finditer('}}', md_clean) ] + \
        # [ x.start() for x in re.finditer('{{', md_clean) ]
added_offset = 0
for i in idxs:
    i += added_offset
    md_clean = md_clean[:i] + "{% raw %}" + md_clean[i:i+2] + "{% endraw %}" + md_clean[i + 2:]
    added_offset += len("{% raw %}{% endraw %}")
with open(output_markdown_abs_path, 'w') as fd:
    fd.write(md_clean)

# Rename .md file to have 'YYYY-MM-DD' prefix that Jekyll expects for all posts
os.rename(output_markdown_abs_path, jekyll_markdown_abs_path)