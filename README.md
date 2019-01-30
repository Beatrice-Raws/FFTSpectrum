FFTSpectrum
===========

A VapourSynth filter that displays the FFT frequency spectrum of a given clip.
Supposedly useful for determining original resolution of upscaled anime content.

Usage
-----

    fftspectrum.FFTSpectrum(clip clip, bint grid=False)

* **clip** - Clip to process. It must have constant format and dimensions, and a luma plane with 8-bit integer samples.
* **grid** - Specifies whether a grid with origin at the center of the image and spacing of 100 pixels should be drawn over the resulting spectrum.

Examples
--------

Without grid:
![No grid](https://user-images.githubusercontent.com/3163182/52003131-cfd4b080-24d4-11e9-9ec8-70c818fce3af.png)

With grid:
![With grid](https://user-images.githubusercontent.com/3163182/52003207-f692e700-24d4-11e9-89d1-c25d5c1617cc.png)

Credits
-------

FFTSpectrum is based on the AviUtl filter with the same name, written by Hiroaki Gotou in 2008.
