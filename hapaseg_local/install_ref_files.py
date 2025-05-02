import sys
import subprocess
import click
from pathlib import Path
from .ref_file_config import HG19_REF_FILES, HG38_REF_FILES

def wget_download(url, local_path):
    try:
        subprocess.run(['wget', '-q', '--show-progress', url, '-O', local_path], check=True)
        print(f"Downloaded file: {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        return None
    
def gsutil_download(gs_uri, local_path):
    print('gsutil attempt')
    subprocess.check_call(f"""gsutil cp -n {gs_uri} {local_path}""", shell = True)

def unzip(local_path):
    subprocess.check_call(f"""gunzip -f {local_path}""", shell = True)

def downlaod_ref_file(source, local_path):
    if source[:3] == 'gs:':
        gsutil_download(source, local_path)
    else:
        wget_download(source, local_path)

    if source[-3:] == '.gz':
        unzip(local_path)


@click.command()
@click.argument('out_dir', type=click.Path())
@click.option('--ref-build', type=str, default='both', help='Reference genome build to download. Options are [hg19, hg38, both]')
def download_ref_files(out_dir, ref_build):
    hg19_dl= hg38_dl = False
    if ref_build =='hg19':
        hg19_dl = True
    elif ref_build =='hg38':
        hg38_dl = True
    elif ref_build =='both':
        hg19_dl = True
        hg38_dl = True
    else:
        raise ValueError(f"Could not interpret build string {ref_build}. Options are [hg19, hg38, both]")

    ref_dir_path = Path(out_dir)
    ref_dir_path.mkdir(exist_ok=True, parents=True)
    
    try:
	    subprocess.check_call("[ -f ~/.config/gcloud/config_sentinel ]", shell = True)
    except subprocess.CalledProcessError:
        print("gcloud is not configured. Please run `gcloud auth login` and `gcloud auth application-default login` and try again.", file = sys.stderr)
        sys.exit(1)

    if hg19_dl:
        print('Downloading reference files for hg19')
        ref_dir_hg19_path = ref_dir_path.joinpath('hg19')
        ref_dir_hg19_path.mkdir(exist_ok=True)

        for file_type, sd_tuple in HG19_REF_FILES.items():
            if file_type == 'ref_1kG':
                ref_1kg_dir = ref_dir_hg19_path.joinpath(sd_tuple[1])
                ref_1kg_dir.mkdir(exist_ok=True)
                results = subprocess.run(f"""bash hg19_download_1kg.sh {ref_1kg_dir} {ref_dir_hg19_path.joinpath(HG19_REF_FILES['ref_fasta'][1])}""",
                                         shell=True
                                         )
            else:
                try:
                    downlaod_ref_file(sd_tuple[0], ref_dir_hg19_path.joinpath(sd_tuple[1]))
                except Exception as e:
                    print(f'Failed download of {file_type} with source {sd_tuple[0]} and local destination {sd_tuple[1]} with error {e}')

    if hg38_dl:
        print('Downloading reference files for hg38')
        ref_dir_hg38_path = ref_dir_path.joinpath('hg38')
        ref_dir_hg38_path.mkdir(exist_ok=True)
        for file_type, sd_tuple in HG38_REF_FILES.items():
            if file_type == 'ref_1kG':
                ref_1kg_dir = ref_dir_hg38_path.joinpath(sd_tuple[1])
                ref_1kg_dir.mkdir(exist_ok=True)
                results = subprocess.run(f"""bash hg38_download_1kg.sh {ref_1kg_dir} {ref_dir_hg38_path.joinpath(HG38_REF_FILES['ref_fasta'][1])}""",
                                            shell=True
                                            )
            else:
                try:
                    downlaod_ref_file(sd_tuple[0], ref_dir_hg38_path.joinpath(sd_tuple[1]))
                except Exception as e:
                    print(f'Failed download of {file_type} with source {sd_tuple[0]} and local destination {sd_tuple[1]} with error {e}')

                if file_type == 'cytoband_file':
                    # need to filter cytoband file to only include autosomes
                    outpath = ref_dir_hg38_path.joinpath(sd_tuple[1])
                    subprocess.check_call(f"""(echo -e "chr\tstart\tend\tband\tstain"; cat {str(outpath).rstrip('.gz')} | grep -E "^(chr([1-9]|1[0-9]|2[0-2]|X|Y))\\b") > tst.txt && mv tst.txt {outpath}""", shell = True)
    print('Done downloading reference files')


if __name__ == '__main__':
    download_ref_files()