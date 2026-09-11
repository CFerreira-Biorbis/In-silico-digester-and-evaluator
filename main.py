# %% Imports

import core
import data_input

# %% Main def


def main():
    source = data_input.get_input()

    if source == 1:
        print("\nYou choose to use accession codes.")
        codes = input("Enter accession codes (comma-separated): ")

        codes = [x.strip() for x in codes.split(",")]

        print(f"You provided: {codes}")
        print("Featching proteins from UniProt, hold on...")

        proteins = data_input.get_uniprot_sequences(codes)

    elif source == 2:
        print("\nYou choose to use supply a FASTA file.")
        filepath = input("Enter the path to the FASTA file: ")

        print(f"Loading {filepath}...")

        proteins = data_input.read_fasta(filepath)

    enzymes = data_input.get_enzymes()

    max_miss = int(
        input("\nHow many missed cleavages you want to test at most? ")
    )

    digestions = []

    for enzyme in enzymes:
        for misses in range(0, max_miss+1):
            for prot in proteins:
                a = core.digest_protein(
                    prot.sequence,
                    enzyme,
                    misses,
                    prot.accession,
                    prot.protein_name,
                )
                digestions.append(a)

    peptides = core.aggregate_peptides(digestions)

    data_df = core.export_as_table(peptides)

    return proteins, enzymes, digestions, peptides, data_df


# %% Main

if __name__ == "__main__":
    proteins, enzymes, digestions, peptides, data_df = main()
    
    print(f'{len(enzymes)} enzymes were used to digest {len(proteins)} proteins.')
    print(f'A total of {len(peptides)} unique peptides were found!')
    
    
