import wolf

class ASCAT(wolf.Task):
    inputs = {'tumor_LogR': None,
              'tumor_BAF':None,
              'normal_LogR': None,
              'normal_BAF':None,
              'GC_correction_file': None,
              'RT_correction_file': None}
    
    script = """
    R -e "library(ASCAT); \
        ascat.bc = ascat.loadData(Tumor_LogR_file = '${tumor_LogR}',\
                                  Tumor_BAF_file = '${tumor_BAF}',\
                                  Germline_LogR_file = '${normal_LogR}', \
                                  Germline_BAF_file = '${normal_BAF}',\
                                  gender = rep('XX', 1), \
                                  genomeVersion = 'hg38'); \
        ascat.bc = ascat.correctLogR(ascat.bc, GCcontentfile = '${GC_correction_file}',\
                                     replictimingfile = '${RT_correction_file}'); \
        ascat.plotRawData(ascat.bc, img.prefix = 'After_correction_'); \
        ascat.bc = ascat.aspcf(ascat.bc); \
        ascat.output = ascat.runAscat(ascat.bc, gamma=1, write_segments = T); \
        QC = ascat.metrics(ascat.bc,ascat.output); \
        write.table(QC, './ascat_QC.txt', quote=FALSE);"
"""
    output_patterns = {
    "ascat_corrected_data_normal" : "After_correction_sim_sample.germline.png",
    "ascat_corrected_data_tumor" : "After_correction_sim_sample.tumour.png",
    "ascat_profile_plot": "sim_sample.ASCATprofile.png",
    "ascat_raw_profile" : "sim_sample.rawprofile.png",
    "ascat_raw_segments" : "sim_sample.segments_raw.txt",
    "ascat_segments" : "sim_sample.segments.txt",
    "ascat_sunrise_plot" : "sim_sample.sunrise.png"
    }
    
    resources = {"cpus-per-task": 4, "mem" : "6G"}
    docker = "gcr.io/broad-getzlab-workflows/ascat:v0"
