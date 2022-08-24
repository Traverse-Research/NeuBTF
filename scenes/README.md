# Mitsuba scenes

The scenes `.xml` can be found in `/scenes`. 

Currently the example scenes are
+ `matpreview_neubtf.xml` 
+ `cloth_neubtf.xml`

To correctly use these scenes: 
In the `/scenes` folders create two folders: 
+ `/envmaps`
+ `/meshes`

Then you have to download some additional files 
+ from [mitsuba's matpreview](https://www.mitsuba-renderer.org/scenes/matpreview.zip):
    + `envmap.exr` to `/envmaps` folder
    + `matpreview.serialized` to `/meshes` folder
+ from [github.com/elerac/btf-rendering](https://github.com/elerac/btf-rendering/tree/master/scenes/cloth) cloth scene:
    + `cloth.obj` to `/meshes` folder
+ [hdrihaven skylit garage](https://hdrihaven.com/files/hdris/skylit_garage_1k.hdr):
    + `skylit_garage_1k.hdr` to `/envmaps` folder
    