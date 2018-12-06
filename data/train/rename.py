# -*- coding: UTF-8 -*-
import os

photo_files = os.popen('dir /b /a-d photos')
sketch_files = os.popen('dir /b /a-d sketches')
a = 1
for photo, sketch in zip(photo_files, sketch_files):
    # print(photo.strip(), sketch.strip())
    photoname = photo.strip()
    sketchname = sketch.strip()
    print(photoname, sketchname)
    if 'jpg' in photoname:
        os.popen('ren photos\\%s %s.jpg' % (photoname, str(a)))
        os.popen('ren sketches\\%s %s.jpg' % (sketchname, str(a)))
        a = a + 1
