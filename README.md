# HapASeg
### Haplotype-Aware Segmentation algorithm for estimating homologue-specific somatic copy number alterations

## Installation
HapASeg requires several dependencies that can be complex to install. We provide a Docker build script, `hapaseg_local/Dockerfile`, to create a container image for running HapASeg locally. (Note: `hapaseg_local/Dockerfile` is distinct from the Dockerfile located in the root directory). 

To build the `hapaseg_local` image, run the following command from the root of the repository:
```bash
docker build -f ./hapaseg_local/Dockerfile -t hapaseg_image .
```

### Reference Files
HapASeg relies on several reference files. These can be downloaded automatically by executing:
```bash
docker run -v {workdir}:/workdir/ hapaseg_image hapaseg_local_install_ref_files --ref-build {ref_build} /workdir/ref_files/
```
* Replace `{ref_build}` with the reference genome build of your input BAMs (`hg19`, `hg38`, or `both`).
* Replace `{workdir}` with the local path where the reference files should be saved. 

**Note:** By default, reference files are downloaded from Zenodo, which may be slow. To skip the download of large reference genome FASTA files, you can replace the file paths in `hapaseg_local/ref_file_config.py` with the paths to your local reference FASTA files.

## Usage
Once the reference files are downloaded, HapASeg can be executed with the following command:
```bash
docker run -v {workdir}:/workdir/ hapaseg_image hapaseg_local [options]
```
Replace `{workdir}` with the local path where your reference files are stored. If your sample BAM files are located outside of the `workdir`, you can mount additional directories using extra `-v` flags. Use the `--help` flag to see all available run options.

### Resource Allocation
`hapaseg_local` includes several subroutines that are **amenable** to parallelization. You can specify the maximum resources you would like HapASeg to use with the `--max-cpus` and `--max-mem` flags. 
* **Default:** HapASeg will attempt to use all available resources. 
* **Minimum Requirement:** The method requires at least **12GB** of memory to run.

## Publication
Details on the method and relevant benchmarking can be found in:
> Priebe *et al.* 2026, *Genome Biology*. DOI: [https://doi.org/10.1186/s13059-026-03971-w](https://doi.org/10.1186/s13059-026-03971-w)

Benchmarking source code for the publication is available in `/benchmarking`.

## WolF
HapASeg is optimized for the **wolF** workflow management platform. The tasks and full workflow are located in `/wolF`.

