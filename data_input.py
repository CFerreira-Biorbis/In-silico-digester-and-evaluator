
# %% Imports

import requests
import core
from core import Protein

# %% Functions


def get_uniprot_sequences(accessions: list[str]) -> list[Protein]:
    """
    Retrieve protein sequences from UniProt.

    Parameters
    ----------
    accessions : list[str]
        UniProt protein accession codes.

    Returns
    -------
    list[Protein]
        Protein objects containing accession and sequence.
    """
    proteins = []

    if not accessions:
        return proteins

    for accession in accessions:

        params = {"fields": ["accession", "protein_name", "sequence"]}
        headers = {"accept": "application/json"}
        base_url = "https://rest.uniprot.org/uniprotkb/" + accession

        response = requests.get(base_url, headers=headers, params=params)
        if not response.ok:
            response.raise_for_status()
            # sys.exit()

        data = response.json()

        proteins.append(
            Protein(
                accession=accession,
                protein_name=data["proteinDescription"]["recommendedName"][
                    "fullName"
                ]["value"],
                sequence=data["sequence"]["value"],
            )
        )

    return proteins


def read_fasta(filename: str) -> list[Protein]:
    """
    Read protein sequences from a FASTA file.

    Each FASTA record is converted into a :class:`Protein` object.
    The accession and protein name are extracted from the FASTA header,
    while the amino-acid sequence is taken from the sequence lines.

    Parameters
    ----------
    filename : str
        Path to the FASTA file to read.

    Returns
    -------
    list[Protein]
        A list of proteins parsed from the FASTA file.

    Raises
    ------
    FileNotFoundError
        If the specified FASTA file does not exist.
    """

    proteins = []

    with open(filename, "r") as f:

        accession = None
        protein_name = None
        sequence_parts = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                # Save the previous protein
                if accession is not None:
                    proteins.append(
                        Protein(
                            accession=accession,
                            protein_name=protein_name,
                            sequence="".join(sequence_parts),
                        )
                    )

                # Parse the new FASTA header
                header = line[1:]

                parts = header.split("|")

                if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
                    accession = parts[1]
                    description = parts[2]
                    protein_code_name = description.split(" OS=", 1)[0]
                    protein_name = " ".join(protein_code_name.split(" ")[1:])

                else:
                    # Generic FASTA header
                    accession = parts[0].split()[0]
                    protein_name = header

                sequence_parts = []

            else:
                sequence_parts.append(line)

        # Don't forget the final protein
        if accession is not None:
            proteins.append(
                Protein(
                    accession=accession,
                    protein_name=protein_name,
                    sequence="".join(sequence_parts),
                )
            )

    return proteins


def get_input():
    """ 
    Initial input intended to give chosing what method of data loading is to
    be used
    """
    
    choice = input(
        "What do you want to provide?\n"
        "1. Accession codes\n"
        "2. FASTA file\n"
        # "3. Direct sequence\n"
        "Choose: "
    )
    
    if int(choice) in [1, 2]:
        return int(choice)
    else:
        raise Exception("Sorry, invalid choice!!")
        
def get_enzymes():
    """
    
    """
    print('\nThe following enzymes are available:')
    core.get_enzymes()
    
    enzymes = input('\nEnter enzymes names (comma-separated): ')
    
    enzymes = [x.strip() for x in enzymes.split(',')]
    
    return enzymes

    
    
    