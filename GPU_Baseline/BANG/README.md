# BANG index builder

`index_build.sh` is a tracked relative link to
`../../script/bang_index_build.sh`, the canonical builder.

The canonical script contains the BANG-specific conversion code. Its external
graph-builder dependency is configured as an ignored local link:

```bash
GPU_Baseline/configure.sh \
  --bang-builder /path/to/build_disk_index
```

For the recorded seven-dataset workflow, use the compatible PipeANN positional
API. The central script also supports the named DiskANN API used by its
Deep1B preset.

Run builds through `../build_index.sh` so the recorded per-dataset graph
parameters and `index/<dataset>/BANG` output location are selected
automatically.
