# Example Gaussian splat assets

The example `.splat` scenes used by `rt_RUSTMIX` are available from the
[cakewalk/splat-data repository on Hugging Face](https://huggingface.co/cakewalk/splat-data).
That repository contains converted Gaussian splats from multiple sources, so
the licensing terms can differ by scene. Review the upstream repository and
source attribution before redistributing any downloaded files.

## Download the complete asset collection

From the project root, use Git:

```powershell
git clone https://huggingface.co/cakewalk/splat-data assets-download
Copy-Item assets-download\*.splat assets\
Remove-Item -Recurse -Force assets-download
```

Or use the Hugging Face CLI:

```powershell
huggingface-cli download cakewalk/splat-data `
  --repo-type model `
  --local-dir assets
```

The application looks for files such as:

```text
assets\garden.splat
assets\treehill.splat
assets\truck.splat
assets\room.splat
assets\train.splat
assets\plush.splat
assets\nike.splat
```

The large binary scene files are intentionally not committed to this Git
repository. The `assets` directory itself is tracked through this document, so
the download location is present after checkout.

Also see the [upstream splat viewer](https://huggingface.co/spaces/cakewalk/splat).
