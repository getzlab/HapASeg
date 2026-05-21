from pathlib import Path

HG19_REF_FILES = dict(
    # fix 1kg ref panel 
        #hg19_ref_dict = pd.read_pickle(CWD + "/ref_panel.hg19.pickle")

        ref_fasta =("https://zenodo.org/records/19325399/files/hg19_Homo_sapiens_assembly19.fasta", "Homo_sapiens_assembly19.fasta"),
        ref_fasta_idx = ("https://zenodo.org/records/19325399/files/hg19_Homo_sapiens_assembly19.fasta.fai", "Homo_sapiens_assembly19.fasta.fai"),
        ref_fasta_dict = ("https://zenodo.org/records/19325399/files/hg19_Homo_sapiens_assembly19.dict", "Homo_sapiens_assembly19.dict"),
        genetic_map_file = ("https://zenodo.org/records/19325399/files/hg19_genetic_map_hg19_withX.txt.gz", "genetic_map_hg19_withX.txt.gz"),
        common_snp_list = ("https://zenodo.org/records/19325399/files/hg19_gnomAD_MAF10_80pct_45prob.txt", "gnomAD_MAF10_80pct_45prob.txt"),
        cytoband_file = ('https://zenodo.org/records/19325399/files/hg19_cytoband.txt', 'cytoband.txt'),
        repl_file = ('https://zenodo.org/records/19325399/files/hg19_RT.raw.hg19.pickle', 'RT.raw.hg19.pickle'),
        faire_file = ('https://zenodo.org/records/19325399/files/hg19_coverage.dedup.raw.10kb.pickle', 'coverage.dedup.raw.10kb.pickle'),
        cfdna_wes_faire_file = ('https://zenodo.org/records/19325399/files/hg19_coverage.w_cfDNA.dedup.raw.10kb.pickle', 'coverage.w_cfDNA.dedup.raw.10kb.pickle'),
        ref_1kG = (None, '1kG')
    )

HG38_REF_FILES = dict(
    # fix 1kg ref panel 
        #hg19_ref_dict = pd.read_pickle(CWD + "/ref_panel.hg19.pickle")

        ref_fasta =("https://zenodo.org/records/19325399/files/hg38_GRCh38.d1.vd1.fa", "hg38_GRCh38.d1.vd1.fa"),
        ref_fasta_idx = ("https://zenodo.org/records/19325399/files/hg38_GRCh38.d1.vd1.fa.fai", "hg38_GRCh38.d1.vd1.fa.fai"),
        ref_fasta_dict = ("https://zenodo.org/records/19325399/files/hg38_GRCh38.d1.vd1.dict", "hg38_GRCh38.d1.vd1.dict"),
        genetic_map_file = ("https://zenodo.org/records/19325399/files/hg38_genetic_map_hg38_withX.txt.gz", "genetic_map_hg38_withX.txt.gz"),
        common_snp_list = ("https://zenodo.org/records/19325399/files/hg38_gnomAD_MAF10_50pct_45prob_hg38_final.txt", "gnomAD_MAF10_50pct_45prob_hg38_final.txt"),
        cytoband_file = ('https://zenodo.org/records/19325399/files/hg38_cytoband.txt', 'cytoband.txt'),
        repl_file = ('https://zenodo.org/records/19325399/files/hg38).raw.hg38.pickle', 'RT.raw.hg38.pickle'),
        faire_file = ('https://zenodo.org/records/19325399/files/hg38_coverage.dedup.raw.10kb.hg38.pickle', 'coverage.dedup.raw.10kb.hg38.pickle'),
        #cfdna_wes_faire_file = ('',)

        ref_1kG = (None, '1kG')
    )

def create_ref_file_dict(ref_path, build_str, override_dict={}):
    
    rp = Path(ref_path)
    
    if build_str == 'hg19':
        ref_file_dict = HG19_REF_FILES
    elif build_str == 'hg38':
        ref_file_dict = HG38_REF_FILES
    else:
        raise ValueError(f"Genome builds supported are hg19 and hg38, recieved build_str {build_str}")

    # load default paths
    ref_path_dict = {k : rp.joinpath(build_str).joinpath(v[1].rstrip('.gz')) for k, v in ref_file_dict.items() }
    # update paths with override_dict
    ref_path_dict = {k : override_dict[k] if k in override_dict else v for k, v in ref_path_dict.items()}
    # check all ref file paths
    for k, v in ref_path_dict.items():
        if not Path(v).exists():
            raise ValueError(f'Path to ref file {k} of {v} could not be found.')
    return ref_path_dict