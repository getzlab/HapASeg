#!/bin/bash
# Script to download hg38 1000 Genomes Phase 3 VCF files for chromosomes 1-22 and X
# and convert them to compressed BCF format with CSI indices

# We then normalize variants using the version of the GRCh38 fasta reference


# Create output directory
OUTDIR=$1
NTHREADS=6
REF_FASTA_PATH=$2
# Define the base URL
BASE_URL="ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
mkdir -p ${OUTDIR}
# Loop through chromosomes 1-22 and X
for chr in {1..22} X; do
  echo "Processing chromosome ${chr}..."
  
  # Define file names
  if [ "${chr}" = "X" ]; then
    VCF_FILE="ALL.chrX.phase3_shapeit2_mvncall_integrated_v1c.20130502.genotypes.vcf.gz"
  else
    VCF_FILE="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
  fi
  VCF_OUT_FILE="${OUTDIR%/}/${VCF_FILE}"
  VCF_INDEX="${VCF_FILE}.tbi"
  OUTPUT_BCF="${OUTDIR%/}/ALL.chr${chr}.high_coverage.bcf"
  
  # Download VCF file if it doesn't exist
  if [ ! -f "${OUTDIR%/}/${VCF_FILE}" ]; then
    echo "Downloading ${VCF_FILE}..."
    wget -q -P ${OUTDIR} --show-progress "${BASE_URL}/${VCF_FILE}"
  else
    echo "${VCF_FILE} already exists, skipping download."
  fi
  
  # Download VCF index file if it doesn't exist
  if [ ! -f "${OUTDIR%/}/${VCF_INDEX}" ]; then
    echo "Downloading ${VCF_INDEX}..."
    wget -q -P ${OUTDIR} --show-progress "${BASE_URL}/${VCF_INDEX}"
  else
    echo "${VCF_INDEX} already exists, skipping download."
  fi

  # Normalize the VCF file
  if [ ! -f "${OUTPUT_BCF}.csi" ]; then
    echo "Normalizing VCF file..."
    set -x
    bcftools view --no-version -Ou -c 2 ${OUTDIR%/}/${VCF_FILE} | \
    bcftools norm --no-version -Ou -m -any | \
    bcftools norm --no-version --threads=${NTHREADS} -Ob -o ${OUTPUT_BCF} -d none -f ${REF_FASTA_PATH} && \
    bcftools index --threads=${NTHREADS} --csi -f ${OUTPUT_BCF} && rm ${OUTDIR%/}/${VCF_FILE} ${OUTDIR%/}/${VCF_INDEX}
  else
    echo "${OUTPUT_BCF}.csi already exists, skipping indexing."
  fi

  echo "Completed processing chromosome ${chr}"
  echo "-------------------------------------"
done

echo "All chromosomes processed successfully!"

# #!/bin/bash
# # Script to download hg19 1000 Genomes Phase 3 VCF files for chromosomes 1-22 and X
# # and convert them to compressed BCF format with CSI indices


# # Create output directory
# OUTPATH=$1
# NTHREADS=6

# # Define the base URL
# BASE_URL="ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"

# # Loop through chromosomes 1-22 and X
# for chr in {1..22} X; do
#   echo "Processing chromosome ${chr}..."
  
#   # Define file names
#   VCF_FILE="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
#   VCF_INDEX="${VCF_FILE}.tbi"
#   OUTPUT_BCF="${OUTPATH}/ALL.chr${chr}.phase3.bcf"
  
#   # Download VCF file if it doesn't exist
#   if [ ! -f "${VCF_FILE}" ]; then
#     echo "Downloading ${VCF_FILE}..."
#     wget -q --show-progress "${BASE_URL}/${VCF_FILE}"
#   else
#     echo "${VCF_FILE} already exists, skipping download."
#   fi
  
#   # Download VCF index file if it doesn't exist
#   if [ ! -f "${VCF_INDEX}" ]; then
#     echo "Downloading ${VCF_INDEX}..."
#     wget -q --show-progress "${BASE_URL}/${VCF_INDEX}"
#   else
#     echo "${VCF_INDEX} already exists, skipping download."
#   fi
  
#   # Convert VCF to BCF and index with CSI
#   echo "Converting to BCF and indexing with CSI..."
#   bcftools view "${VCF_FILE}" -Ob -l 4 --threads=${NTHREADS} -o "${OUTPUT_BCF}"
#   bcftools index --threads=${NTHREADS} --csi "${OUTPUT_BCF}"
  
#   echo "Completed processing chromosome ${chr}"
#   echo "-------------------------------------"
# done

# echo "All chromosomes processed successfully!"