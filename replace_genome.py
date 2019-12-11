from Bio import SeqIO
import numpy as np

genomeSeq = SeqIO.parse("/home/malab14/research/Genomes/Arabidopsis/TAIR10_chr_all.fas", "fasta")

genomeSeqDict = {}
for fasta in genomeSeq:
    name, sequence = fasta.id, fasta.seq.tostring()
    name = "Chr" + str(name)
    print(name)
    genomeSeqDict[name] = sequence

# SNP = np.loadtxt("/home/malab14/research/ArabidopsisGWAS/00datasets/genotypes/1001Genomes/final_snp_allel.txt",
#                  dtype="str")
# Chr = np.core.defchararray.split(SNP[:, 0], "_")
# Chr = np.vstack(Chr)
# SNP = np.column_stack((SNP, Chr))

tt = 0
with open("/home/malab14/research/ArabidopsisGWAS/00datasets/genotypes/1001Genomes/final_snp_allel.txt") as f:
    for line in f:
        line = line.strip().split("\t")
        curChr = line[0].split("_")[0]
        curPos = int(line[0].split("_")[1])
        curSeq = list(genomeSeqDict[curChr])
        curSeq[curPos-1] = line[2]
        curSeq = ''.join(curSeq)
        genomeSeqDict[curChr] = ''.join(curSeq)
        tt = tt + 1
        print(tt)
        #break

f.close()