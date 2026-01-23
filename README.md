# HapASeg
### Haplotype Aware Segmentation algorithm for estimating homologue-specific somatic copy number alterations

## Installation
HapASeg requires a number of dependencies that can be complicated to install. We provide a docker image that builds a container for running HapASeg in `hapaseg_local/Dockerfile`. To build the container from the root directory, invoke ```docker build -f ./hapaseg_local/Dockerfile -t hapaseg_image .``` 

HapASeg also relies on a number of reference files. These files can be downloaded automatically by calling ```docker run -v {./workdir}:{/workdir/} hapaseg_image hapaseg_local_install_ref_files --ref-build {ref-build} /workdir/ref_files/``` where `ref_build` is the reference genome build of the input sample BAMS (hg19 or hg38 or both) and `workdir` is the local path where the reference files will be saved. 

## Usage
Once the reference files have been downloaded, the HapASeg can be executed by calling ```docker run -v {workdir}:{/workdir/} hapaseg_image hapaseg_local ...``` with the desired inputs. Use `--help` to see the run options. Here, `workdir` is the local path where the reference files were saved. Directories containing sample BAM files can be mounted with additional -v mount commands if they are not already in `workdir`.

`hapaseg_local` has several subroutines that are ammenable to parallelization. Set the maximum number of cpus and memory that you would like HapASeg to using the `--max-cpus` and `--max-mem` commands. The default behaviour is to use all available resources. The method requires at least 12GB of memory to run.

## Publication
Details on the method and relevant benchmarking can be found in our preprint XXX with citation XXX.

## WolF
HapASeg has been optimized for running on the wolF workflow managment platform. The tasks and full workflow can be found in `/wolF`. Benchmarking for the aforementioned publication was also done in wolF with source code available in `/benchmarking`. 