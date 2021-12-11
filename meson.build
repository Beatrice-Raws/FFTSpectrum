project('FFTSpectrum', 'c',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'c_std=c99'],
  meson_version : '>=0.49.0',
  version : '2'
)

sources = 'FFTSpectrum.c'

compiler = meson.get_compiler('c')

if compiler.get_argument_syntax() == 'msvc'
  deps = [ dependency('fftwf') ]
  install_dir = 'installed'  # dummy
else
  vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)
  deps = [ dependency('fftw3f'), vapoursynth_dep ]
  install_dir = join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth')
endif

shared_module('fftspectrum', sources,
  dependencies : deps,
  install : true,
  install_dir : install_dir,
  gnu_symbol_visibility : 'hidden'
)
