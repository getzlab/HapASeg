import wolf
import create_1kG_PoN
import pandas as pd
import os
import subprocess 

# WGS 1kG GATK PoN
#paths_df = pd.read_csv(./'PCR_free_WGS_1kG_random_50_samples.tsv', sep='\t', header=None)
#cram_paths = paths_df.iloc[:,0].values
#sample_names = paths_df.iloc[:,14].values
#crai_paths = cram_paths + '.crai'
#
#with wolf.Workflow(workflow = create_1kG_PoN.gatk_make_pon_workflow) as w:
#        w.run(run_name = 'gatk_1kg_pon',
#              cram_files = list(cram_paths), crai_files = list(crai_paths),
#              sample_names = list(sample_names),
#              ref_genome = 'gs://genomics-public-data/resources/broad/hg38/v0/Homo_sapiens_assembly38.fasta',
#              ref_index = 'gs://genomics-public-data/resources/broad/hg38/v0/Homo_sapiens_assembly38.fasta.fai',
#              ref_dict = 'gs://genomics-public-data/resources/broad/hg38/v0/Homo_sapiens_assembly38.dict',
#              interval_list = './wgs_hg38_1kb_gatk.interval_list',
#              annotation_file = './wgs_hg38_1kb_gatk_reformatted_dict.annotated_intervals.tsv')

sub_out = subprocess.check_output('gsutil ls gs://jh-xfer/realignment/NA12878_TWIST_*/*.bam', shell=True)
twist_bams = sub_out.decode().split()
twist_bams = [b for b in twist_bams if not '36' in b] # remove the primary
twist_bais = [b + '.bai' for b in twist_bams]
twist_names = [os.path.basename(b)[:-4] for b in twist_bams]

# TWIST hg38 samples
# ref genomes are only different in decoys, since GATK is very picky
with wolf.Workflow(workflow = create_1kG_PoN.gatk_make_pon_workflow) as w:
        w.run(run_name = 'gatk_twist_pon',
              cram_files = twist_bams, crai_files = twist_bais,
              sample_names = twist_names,
              ref_genome = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa',
              ref_index = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.fa.fai',
              ref_dict = 'gs://getzlab-workflows-reference_files-oa/hg38/gdc/GRCh38.d1.vd1.dict',
              pon_name = 'twist_hg38_PoN',
              interval_list = 'gs://opriebe-tmp/HapASeg/benchmarking/NA12878_exome/broad_custom_exome_v1.Homo_sapiens_assembly38.targets.autosome.interval_list'
             )
