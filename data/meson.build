desktop_file = i18n.merge_file(
  input: 'com.schelb.summarizing.desktop.in',
  output: 'com.schelb.summarizing.desktop',
  type: 'desktop',
  po_dir: '../po',
  install: true,
  install_dir: join_paths(get_option('datadir'), 'applications')
)

desktop_utils = find_program('desktop-file-validate', required: false)
if desktop_utils.found()
  test('Validate desktop file', desktop_utils,
    args: [desktop_file]
  )
endif

appstream_file = i18n.merge_file(
  input: 'com.schelb.summarizing.appdata.xml.in',
  output: 'com.schelb.summarizing.appdata.xml',
  po_dir: '../po',
  install: true,
  install_dir: join_paths(get_option('datadir'), 'appdata')
)

appstream_util = find_program('appstream-util', required: false)
if appstream_util.found()
  test('Validate appstream file', appstream_util,
    args: ['validate', appstream_file]
  )
endif

install_data('com.schelb.summarizing.gschema.xml',
  install_dir: join_paths(get_option('datadir'), 'glib-2.0/schemas')
)

install_data(['model/config.json',
	      'model/gitattributes',
	      'model/pytorch_model.bin',
	      'model/README.md',
	      'model/spiece.model',
	      'model/tokenizer.json'],
  install_dir: join_paths(get_option('datadir'), 'model')
)

compile_schemas = find_program('glib-compile-schemas', required: false)
if compile_schemas.found()
  test('Validate schema file', compile_schemas,
    args: ['--strict', '--dry-run', meson.current_source_dir()]
  )
endif

subdir('icons')
