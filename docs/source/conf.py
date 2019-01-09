html_show_sourcelink = False
project = 'Transit Least Squares'
copyright = '2019 Michael Hippke, Rene Heller'
author = 'Michael Hippke, Rene Heller'
version = ''
release = ''
extensions = ['sphinx.ext.mathjax',]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = None
exclude_patterns = []
pygments_style = None
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
htmlhelp_basename = 'TLSdoc'
latex_elements = {
}
latex_documents = [
    (master_doc, 'TLS.tex', 'TLS Documentation',
     'Michael Hippke, Rene Heller', 'manual'),
]
man_pages = [
    (master_doc, 'tls', 'TLS Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'TLS', 'TLS Documentation',
     author, 'TLS', 'One line description of project.',
     'Miscellaneous'),
]
epub_title = project
epub_exclude_files = ['search.html']