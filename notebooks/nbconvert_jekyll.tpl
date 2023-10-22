{% extends 'markdown/index.md.j2' %}

{% block data_svg %} 
![svg]({{ output.metadata.filenames['image/svg+xml'] | path2support }}) 
{% endblock data_svg %} 

{% block data_png %} 
![png]({{ output.metadata.filenames['image/png'] | path2support }}) 
{% endblock data_png %} 

{% block data_jpg %} 
![jpeg]({{ output.metadata.filenames['image/jpeg'] | path2support }}) 
{% endblock data_jpg %} 