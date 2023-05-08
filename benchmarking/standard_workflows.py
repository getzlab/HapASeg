import wolf
from wolf import LocalizeToDisk
from ascat_wolf_task.ascat_wolf_task import *
from facets_wolf_task.facets_wolf_task import *
from gatk_wolf_task.gatk_standard_workflow import *
from hatchet_wolf_task.workflow import *
from output_comparison_tasks import *

mutect1 = wolf.ImportTask(
  task_path = "git@github.com:getzlab/MuTect1_TOOL.git",
  main_task = "mutect1",
  commit = "74df599"
)

hapaseg = wolf.ImportTask(
    task_path='../'
    )

def Facets_ASCAT_Standard(
                    tumor_bam=None,
                    tumor_bai=None,
                    normal_bam=None,
                    normal_bai=None,
                    sample_name=None,
                    ref_fasta=None,
                    ref_fasta_idx=None,
                    ref_fasta_dict=None,
                    common_snp_list=None,
                    cytoband_file=None,
                    localization_token=None,
                    persistent_disk_dry_run=False,
                    ASCAT_GC_correction_file=None,
                    ASCAT_RT_correction_file=None     
               ):
    """
    run these methods together since we can use the same allelecounts
    """

    # localize_bams
    tumor_loc_task = LocalizeToDisk(
          files = {
            "t_bam" : tumor_bam,
            "t_bai" : tumor_bai,
          },
        token=localization_token,
        persistent_disk_dry_run = persistent_disk_dry_run
        )

    normal_loc_task = LocalizeToDisk(
          files = {
            "n_bam" : normal_bam,
            "n_bai" : normal_bai
          },
        token=localization_token,
        persistent_disk_dry_run = persistent_disk_dry_run
        )
    
    t_bam = tumor_loc_task['t_bam']
    t_bai = tumor_loc_task['t_bai']
    n_bam = normal_loc_task['n_bam']
    n_bai = normal_loc_task['n_bai']
    
    # localize ref files
    localization_task = LocalizeToDisk(
      files = dict(
        ref_fasta = ref_fasta,
        ref_fasta_idx = ref_fasta_idx,
        ref_fasta_dict = ref_fasta_dict,
        common_snp_list = common_snp_list,
        ASCAT_GC_correction_file = ASCAT_GC_correction_file,
        ASCAT_RT_correction_file=ASCAT_RT_correction_file
        )
    )

    split_het_sites = wolf.Task(
          name = "split_het_sites",
          inputs = { "snp_list" : localization_task["common_snp_list"] },
          script = """
          grep '^@' ${snp_list} > header
          sed '/^@/d' ${snp_list} | split -l 50000 -d -a 4 --filter 'cat header /dev/stdin > $FILE' --additional-suffix '.picard' - snp_list_chunk
          """,
          outputs = { "snp_list_shards" : "snp_list_chunk*" }
        )    

    m1_task = mutect1(inputs=dict(
      pairName = "het_coverage",
      caseName = "tumor",
      ctrlName = "normal",

      t_bam = tumor_loc_task["t_bam"],
      t_bai = tumor_loc_task["t_bai"],
      n_bam = normal_loc_task["n_bam"],
      n_bai = normal_loc_task["n_bai"],

      fracContam = 0,

      refFasta = localization_task["ref_fasta"],
      refFastaIdx = localization_task["ref_fasta_idx"],
      refFastaDict = localization_task["ref_fasta_dict"],

      intervals = split_het_sites["snp_list_shards"],

      exclude_chimeric = True,

      force_calling = True,
    ))

    m1_gather = wolf.Task(
      name = "m1_gather",
      inputs = { "callstats_array" : [m1_task["mutect1_cs"]],
                 "sample_name" : sample_name },
      script = """
      head -n2 $(head -n1 ${callstats_array}) > header
      while read -r i; do
        sed '1,2d' $i
      done < ${callstats_array} | sort -k1,1V -k2,2n > cs_sorted
      cat header cs_sorted > ${sample_name}_mutect_callstats.tsv
      """,
      outputs = { "cs_gather" : "*mutect_callstats.tsv" }
    )

    # preprocess raw data
    
    ## facets
    facets_preprocess = Facets_convert_mutect_callstats (inputs = {"callstats_file":m1_gather['cs_gather'],
                                               "sample_name": sample_name})
    ## ascat
    ascat_preprocess= ASCAT_convert_mutect_callstats (inputs = {"callstats_file":m1_gather['cs_gather'],
                                               "sample_name": sample_name})

    #run methods
    run_facets = Facets(inputs = {'snp_counts': facets_preprocess['facets_input_file']})

    run_ascat = ASCAT(inputs = {'tumor_LogR': ascat_preprocess['tumor_LogR'],
                                'tumor_BAF': ascat_preprocess['tumor_BAF'],
                                'normal_LogR': ascat_preprocess['normal_LogR'],
                                'normal_BAF': ascat_preprocess['normal_BAF'],
                                'GC_correction_file': localization_task['ASCAT_GC_correction_file'],
                                'RT_correction_file': localization_task['ASCAT_RT_correction_file']
                                }
                     )

    # plot results
    plot_ascat = Standard_ASCAT_Plotting(inputs = {"ascat_seg_file": run_ascat['ascat_raw_segments'],
                                                   "sample_name": sample_name,
                                                   "ref_fasta":  localization_task["ref_fasta"],
                                                   "cytoband_file": cytoband_file
                                                  }
                                       )

    plot_facets = Standard_Facets_Plotting(inputs ={'facets_seg_file': run_facets['facets_seg_file'],
                                                   "sample_name": sample_name,
                                                   "ref_fasta":  localization_task["ref_fasta"],
                                                   "cytoband_file": cytoband_file
                                                   }
                                          )

def run_all_standard_pipelines(
                    tumor_bam=None,
                    tumor_bai=None,
                    normal_bam=None,
                    normal_bai=None,
                    sample_name=None,
                    ref_fasta=None,
                    ref_fasta_idx=None,
                    ref_fasta_dict=None,
                    reference_genome_version="hg38",
                    cytoband_file=None,
                    localization_token=None,
                    persistent_disk_dry_run=False,
                    # hapaseg
                    hapaseg_single_ended=False,
                    hapaseg_target_list=None,
                    hapaseg_is_ffpe=False,
                    hapaseg_is_cfdna=False,
                    # gatk
                    gatk_common_snp_vcf=None,
                    gatk_interval_list=None,
                    gatk_interval_name=None,
                    gatk_exclude_sex=False,
                    gatk_count_panel=None,
                    gatk_exclude_chroms="",
                    gatk_bin_length=1000,
                    gatk_padding=250,
                    # hatchet
                    hatchet_phase_snps=True,
                    hatchet_common_snp_vcf=None,
                    hatchet_common_snp_vcf_idx=None,
                    hatchet_chromosomes="",
                    # facets ascat
                    fascat_common_snp_list=None,
                    ASCAT_GC_correction_file=None,
                    ASCAT_RT_correction_file=None,
                    upload_bucket=None   
        ):
    # localize bams once
    tumor_loc_task = LocalizeToDisk(
          files = {
            "t_bam" : tumor_bam,
            "t_bai" : tumor_bai,
          },
        token=localization_token,
        persistent_disk_dry_run = persistent_disk_dry_run
        )

    normal_loc_task = LocalizeToDisk(
          files = {
            "n_bam" : normal_bam,
            "n_bai" : normal_bai
          },
        token=localization_token,
        persistent_disk_dry_run = persistent_disk_dry_run
        )
    
    # localize seg plotting inputs seperately for hapaseg
    seg_plot_loc = LocalizeToDisk(
        files = {
            'ref_fasta': ref_fasta,
            'ref_fasta_idx': ref_fasta_idx,
            'cytoband_file': cytoband_file
                }
        )
    
    t_bam = tumor_loc_task['t_bam']
    t_bai = tumor_loc_task['t_bai']
    n_bam = normal_loc_task['n_bam']
    n_bai = normal_loc_task['n_bai']
    
    hapaseg_pipeline = hapaseg.hapaseg_workflow(
                          tumor_bam = t_bam,
                          tumor_bai = t_bai,
                          normal_bam = n_bam,
                          normal_bai = n_bai,
                          single_ended = hapaseg_single_ended,
                          ref_genome_build=reference_genome_version,
                          target_list = hapaseg_target_list,
                          localization_token=localization_token,

                          persistent_dry_run = persistent_disk_dry_run,
                          cleanup_disks=False,
                          is_ffpe = hapaseg_is_ffpe,
                          is_cfdna = hapaseg_is_cfdna)

    # create basic hapaseg seg plot
    hapaseg_seg_plot = Standard_HapASeg_Plotting(inputs={'hapaseg_seg_file': hapaseg_pipeline['hapaseg_segfile'],
                                                         'ref_fasta': seg_plot_loc['ref_fasta'],
                                                         'sample_name': sample_name,
                                                         'cytoband_file': seg_plot_loc['cytoband_file']}
                                                )

    gatk_pipeline = GATK_standard_pipeline(
                            tumor_bam=t_bam,
                            tumor_bai=t_bai,
                            normal_bam=n_bam,
                            normal_bai=n_bai,
                            vcf_file=gatk_common_snp_vcf, 
                            ref_fasta=ref_fasta,
                            ref_fasta_idx=ref_fasta_idx,
                            ref_fasta_dict=ref_fasta_dict,
                            cytoband_file=seg_plot_loc['cytoband_file'],
                            interval_list=gatk_interval_list,
                            interval_name=gatk_interval_name,
                            sample_name=sample_name,
                            bin_length=gatk_bin_length,
                            padding=gatk_padding,
                            exclude_sex=gatk_exclude_sex,
                            upload_bucket=upload_bucket,
                            exclude_chroms=gatk_exclude_chroms,
                            count_panel=gatk_count_panel,
                            localization_token=localization_token,
                            persistent_disk_dry_run=persistent_disk_dry_run
                            )

    hatchet_pipeline =  Hatchet_Run_Standard(
                       sample_name = sample_name,
                       ref_fasta=ref_fasta,
                       ref_fasta_idx=ref_fasta_idx,  
                       ref_fasta_dict=ref_fasta_dict,
                       reference_genome_version=reference_genome_version,
                       tumor_bam = t_bam,
                       tumor_bai = t_bai,
                       normal_bam = n_bam,
                       normal_bai = n_bai,
                       cytoband_file=seg_plot_loc['cytoband_file'],
                       common_snp_vcf=hatchet_common_snp_vcf,
                       common_snp_vcf_idx=hatchet_common_snp_vcf_idx,
                       sample_names = f"{sample_name}_normal {sample_name}_tumor",
                       chromosomes = hatchet_chromosomes,
                       phase_snps = hatchet_phase_snps,
                       upload_bucket = upload_bucket,
                       localization_token=localization_token,
                       persistent_disk_dry_run=persistent_disk_dry_run
                       )

        # run fascets and ASCAT
    fascat_pipeline = Facets_ASCAT_Standard(
                        tumor_bam=t_bam,
                        tumor_bai=t_bai,
                        normal_bam=n_bam,
                        normal_bai=n_bai,
                        sample_name=sample_name,
                        ref_fasta=ref_fasta,
                        ref_fasta_idx=ref_fasta_idx,
                        ref_fasta_dict=ref_fasta_dict,
                        cytoband_file=seg_plot_loc['cytoband_file'],
                        common_snp_list=fascat_common_snp_list,
                        localization_token=localization_token,
                        persistent_disk_dry_run=persistent_disk_dry_run,
                        ASCAT_GC_correction_file=ASCAT_GC_correction_file,
                        ASCAT_RT_correction_file=ASCAT_RT_correction_file     
                   )


# standard plotting only

def standard_plotting_pipeline(sample_name,
                               ref_fasta,
                               ref_fasta_idx,
                               cytoband_file,
                               hapaseg_segfile,
                               ascat_segfile,
                               facets_segfile,
                               gatk_segfile,
                               hatchet_bins,
                               hatchet_segments):

    
    # localize seg plotting inputs seperately for hapaseg
    seg_plot_loc = LocalizeToDisk(
        files = {
            'ref_fasta': ref_fasta,
            'ref_fasta_idx': ref_fasta_idx,
            'cytoband_file': cytoband_file
                }
        )

    hapaseg_seg_plot = Standard_HapASeg_Plotting(inputs={'hapaseg_seg_file': hapaseg_segfile,
                                                         'ref_fasta': seg_plot_loc['ref_fasta'],
                                                         'sample_name': sample_name,
                                                         'cytoband_file': seg_plot_loc['cytoband_file']}
                                                )
    ascat_seg_plot = Standard_ASCAT_Plotting(inputs={'ascat_seg_file': ascat_segfile,
                                                         'ref_fasta': seg_plot_loc['ref_fasta'],
                                                         'sample_name': sample_name,
                                                         'cytoband_file': seg_plot_loc['cytoband_file']}
                                                )
    facets_seg_plot = Standard_Facets_Plotting(inputs={'facets_seg_file': facets_segfile,
                                                         'ref_fasta': seg_plot_loc['ref_fasta'],
                                                         'sample_name': sample_name,
                                                         'cytoband_file': seg_plot_loc['cytoband_file']}
                                                )
    
    GATK_seg_plot = Standard_GATK_Plotting(inputs={'gatk_seg_file': gatk_segfile,
                                                         'ref_fasta': seg_plot_loc['ref_fasta'],
                                                         'sample_name': sample_name,
                                                         'cytoband_file': seg_plot_loc['cytoband_file']}
                                                )

    hatchet_seg_plot = Standard_Hatchet_Plotting(inputs={'hatchet_bin_file': hatchet_bins,
                                                         'hatchet_seg_file': hatchet_segments,
                                                         'ref_fasta': seg_plot_loc['ref_fasta'],
                                                         'sample_name': sample_name,
                                                         'cytoband_file': seg_plot_loc['cytoband_file']}
                                                )
    
