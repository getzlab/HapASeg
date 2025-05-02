# HapASeg
### A haplotype aware segmentation algorithm for calling homologue specific somatic copy number events

## Installation
HapASeg requires a number of dependencies that can be complicated to install. We provide a docker image that builds a container for running HapASeg in `hapaseg_local/Dockerfile`. One can build the container from the root directory by invoking ```docker build -f ./hapaseg_local/Dockerfile -t hapaseg_image .``` 
HapASeg also relies on a number of reference files that can be downloaded automatically by calling ```docker run -v {workdir}:{/workdir/} hapaseg_image hapaseg_local_install_ref_files /workdir/ref_files/``` where `workdir` is the local path with the necessary input files, which will be mounted to `/workdir/` within the container. Adjust filepaths arguments accordingly. 

## Usage
Once the reference files have been downloaded, the method can be run by calling ```docker run -v {workdir}:{/workdir/} hapaseg_image hapaseg_local ...``` with the desired inputs. Use `--help` for more options.

`hapaseg_local` has several subroutines that are ammenable to parallelization. Set the maximum number of cpus and memory that you would like HapASeg to using the `--max-cpus` and `--max-mem` commands. The default behaviour is to use all available resources. The method requires at least 12GB of memory to run.

## Publication
Details on the method and relevant benchmarking can be found in our preprint XXX with citation XXX.

## WolF
HapASeg has been optimized for running on the wolF workflow managment platform. The tasks and full workflow can be found in `/wolF`. Benchmarking for the aforementioned publication was also done in wolF with source code available in `/benchmarking`. 