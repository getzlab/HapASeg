#!/bin/bash
# Script to download hg38 1000 Genomes Phase 3 VCF files for chromosomes 1-22 and X
# and convert them to compressed BCF format with CSI indices

# We then normalize variants using the version of the GRCh38 fasta reference


# Create output directory
OUTDIR=$1
NTHREADS=6
REF_FASTA_PATH=$2
# Define the base URL
BASE_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"

# Loop through chromosomes 1-22 and X
for chr in {1..22} X; do
  echo "Processing chromosome ${chr}..."
  
  # Define file names
  if [ "${chr}" = "X" ]; then
    VCF_FILE="1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz"
  else
    VCF_FILE="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
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
    (bcftools view --no-version -h ${VCF_OUT_FILE} | grep -v "^##contig=<ID=[GNh]" | sed 's/^##contig=<ID=MT/##contig=<ID=chrM/;s/^##contig=<ID=\([0-9XY]\)/##contig=<ID=chr\1/'; bcftools view --no-version -H ${VCF_OUT_FILE} -c 2) | \
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
