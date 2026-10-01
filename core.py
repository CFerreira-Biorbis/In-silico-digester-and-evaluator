"""
Utilities for working with proteins, peptides, and enzymatic digestion.

This module provides data classes and helper functions for common
proteomics workflows, including reading protein sequences from FASTA
files, retrieving protein sequences and metadata from UniProt,
performing enzymatic protein digestion with PyOpenMS, aggregating
peptides from multiple digestions, and exporting peptide information
as a pandas DataFrame.

The main data classes are:

    :class:Protein — represents a protein sequence and associated
    UniProt metadata, with properties for sequence length, molecular
    weight, monoisotopic mass, and amino-acid composition.
    :class:Peptide — represents a peptide sequence and its associated
    source proteins/digestions.
    :class:PeptideSource — describes the protein and digestion
    conditions from which a peptide originated.
    :class:PeptideDigest — represents the result of digesting a protein
    sequence with a specified proteolytic enzyme.

The module uses PyOpenMS for molecular-weight calculations and
enzymatic digestion, requests for UniProt REST API access, and pandas
for tabular export.
"""

# %% Imports

from dataclasses import dataclass, field
import re
from collections import Counter
import pandas as pd

import pyopenms as oms

# %% Dataclasses


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

@dataclass
class AminoacidSequence:
    
    
    sequence: str

    def __post_init__(self) -> None:

        self.sequence = self.sequence.upper().strip().replace(" ", "")

        invalid_chars = set(self.sequence) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"Invalid amino acid residue(s) found in sequence: {sorted(invalid_chars)}. "
                f"Sequence must contain only standard single-letter codes: {sorted(VALID_AMINO_ACIDS)}"
            )

    @property
    def is_empty(self) -> bool:
        """
        Return ``True`` if the protein sequence is empty.
        """
        return not bool(self.sequence)

    @property
    def n_terminal_residue(self) -> str | None:
        """
        Return the amino-acid residue at the N-terminus.

        Returns
        -------
        str or None
            The first residue in the sequence, or ``None`` if the
            sequence is empty.
        """

        return self.sequence[0] if self.sequence else None

    @property
    def c_terminal_residue(self) -> str | None:
        """
        Return the amino-acid residue at the C-terminus.

        Returns
        -------
        str or None
            The last residue in the sequence, or ``None`` if the
            sequence is empty.
        """
        return self.sequence[-1] if self.sequence else None

    @property
    def number_of_residues(self) -> int:
        """
        Return the number of amino-acid residues in the protein.
        """
        return len(self.sequence)

    @property
    def molecular_weight(self) -> float:
        """
        Calculate the average molecular weight of the protein in Da.

        Returns
        -------
        float
            Average molecular weight in Daltons. Returns ``0.0`` for
            an empty sequence.
        """
        if not self.sequence:
            return 0.0

        return oms.AASequence.fromString(self.sequence).getAverageWeight()

    @property
    def monoisotopic_mass(self) -> float:
        """
        Calculate the monoisotopic mass of the protein in Da.

        Returns
        -------
        float
            Monoisotopic mass in Daltons. Returns ``0.0`` for an empty
            sequence.
        """
        if not self.sequence:
            return 0.0

        return oms.AASequence.fromString(self.sequence).getMonoWeight()

    @property
    def aminoacids_counts(self) -> dict[str, int]:
        """
        Return the absolute count of each amino-acid residue.

        Returns
        -------
        dict[str, int]
            Dictionary mapping each residue observed in the sequence
            to its absolute count.
        """
        return dict(Counter(self.sequence))

    @property
    def aminoacids_frequencies(self) -> dict[str, float]:
        """
        Return the relative frequency of each amino-acid residue.

        Returns
        -------
        dict[str, float]
            Dictionary mapping each residue observed in the sequence
            to its relative frequency, expressed as a fraction between
            0 and 1. Returns an empty dictionary for an empty sequence.
        """
        if not self.sequence:
            return {}

        counts = Counter(self.sequence)
        length = len(self.sequence)

        return {aa: count / length for aa, count in counts.items()}

    @property
    def bcaa_count(self) -> int:
        """
        Return the number of branched-chain amino-acid (BCAA) residues.

        BCAAs are leucine (L), isoleucine (I), and valine (V).
        """
        return sum(self.sequence.count(aa) for aa in "LIV")

    @property
    def aromatic_count(self) -> int:
        """
        Return the number of aromatic amino-acid residues.

        Aromatic residues considered are phenylalanine (F),
        tryptophan (W), and tyrosine (Y).
        """
        return sum(self.sequence.count(aa) for aa in "FWY")

    @property
    def bcaa_fraction(self) -> float:
        """
        Return the fraction of residues that are BCAAs.

        Returns
        -------
        float
            Fraction of the sequence composed of leucine, isoleucine,
            and valine, between 0 and 1. Returns ``0.0`` for an empty
            sequence.
        """
        return (
            self.bcaa_count / self.number_of_residues if self.sequence else 0.0
        )

    @property
    def aromatic_fraction(self) -> float:
        """
        Return the fraction of residues that are aromatic amino acids.

        Returns
        -------
        float
            Fraction of the sequence composed of phenylalanine,
            tryptophan, and tyrosine, between 0 and 1. Returns ``0.0``
            for an empty sequence.
        """
        return (
            self.aromatic_count / self.number_of_residues
            if self.sequence
            else 0.0
        )

    def count_residue(self, amino_acid: str) -> int:
        """
        Return the number of occurrences of a specified amino acid.

        Parameters
        ----------
        amino_acid : str
            One-letter amino-acid code to count.

        Returns
        -------
        int
            Number of occurrences of the specified amino acid.
        """
        return self.sequence.count(amino_acid)

    def count_residue_relative(self, amino_acid: str) -> float:
        """
        Return the relative frequency of a specified amino acid.

        Parameters
        ----------
        amino_acid : str
            One-letter amino-acid code to count.

        Returns
        -------
        float
            Fraction of residues represented by the specified amino acid.
        """
        return (
            self.sequence.count(amino_acid) / len(self.sequence)
            if self.sequence
            else 0.0
        )
@dataclass(frozen=True)
class PeptideSource:
    """
    Information describing the source of a peptide.

    Stores the protein and digestion information associated with a
    peptide, without storing the peptide sequence itself.

    Parameters
    ----------
    protein_sequence : str
        Amino-acid sequence of the source protein.
    enzyme : str
        Name of the proteolytic enzyme used to generate the peptide.
    missed_cleavages : int
        Number of missed cleavages allowed during digestion.
    accession : str | None, optional
        Accession identifier of the source protein, if available.
    protein_name : str | None, optional
        Name of the source protein, if available.
    """

    protein_sequence: str
    enzyme: str
    missed_cleavages: int
    accession: str | None = None
    protein_name: str | None = None


@dataclass
class Protein(AminoacidSequence):
    """
    Represent a protein sequence and associated metadata.

    The class stores the protein identifier, name, and amino-acid sequence,
    and provides derived properties describing the sequence composition,
    terminal residues, molecular mass, and selected amino-acid groups.

    Molecular weight and monoisotopic mass are calculated using OpenMS.

    Parameters
    ----------
    accession : str
        Protein accession identifier, such as a UniProt accession.
    name : str
        Protein name or description.
    sequence : str
        Protein amino-acid sequence using one-letter amino-acid codes.

    Properties
    ----------
    is_empty : bool
        Whether the protein sequence is empty.
    n_terminal_residue : str or None
        Amino-acid residue at the N-terminus of the sequence.
        Returns ``None`` for an empty sequence.
    c_terminal_residue : str or None
        Amino-acid residue at the C-terminus of the sequence.
        Returns ``None`` for an empty sequence.
    number_of_residues : int
        Number of amino-acid residues in the sequence.
    molecular_weight : float
        Average molecular weight of the protein in Daltons (Da).
    monoisotopic_mass : float
        Monoisotopic mass of the protein in Daltons (Da).
    aminoacids_counts : dict[str, int]
        Absolute count of each amino-acid residue observed in the sequence.
        Only residues present in the sequence are included.
    aminoacids_frequencies : dict[str, float]
        Relative frequency of each amino-acid residue observed in the
        sequence, expressed as a fraction between 0 and 1.
    bcaa_count : int
        Number of branched-chain amino-acid (BCAA) residues
        (leucine, isoleucine, and valine) in the sequence.
    aromatic_count : int
        Number of aromatic amino-acid residues
        (phenylalanine, tryptophan, and tyrosine) in the sequence.
    bcaa_fraction : float
        Fraction of residues that are BCAAs, expressed as a value
        between 0 and 1.
    aromatic_fraction : float
        Fraction of residues that are aromatic amino acids, expressed
        as a value between 0 and 1.

    Methods
    -------
    count_residue(amino_acid)
        Return the number of occurrences of a specified amino acid.
    count_residue_relative(amino_acid)
        Return the relative frequency of a specified amino acid.

    Notes
    -----
    The sequence is expected to use standard one-letter amino-acid codes.
    Validation of the sequence contents is not currently performed.
    """

    accession: str = "No code"
    name: str = "No name"

    

    def digest(self, enzyme: str, missed_cleavages: int = 0):
        """
        Digest a protein sequence into peptides.

        Parameters
        ----------
        enzyme : str
            Name of the proteolytic enzyme used for digestion.
        missed_cleavages : int
            Maximum number of allowed missed cleavages.

        Returns
        -------
        PeptideDigest
            Digested peptides together with the digestion parameters.
        """

        return digest_protein(
            self, enzyme=enzyme, missed_cleavages=missed_cleavages
        )

    def __repr__(self) -> str:
        return (
            f"Protein("
            f"accession = '{self.accession}', "
            f"name = '{self.name}', "
            f"sequence_length = {self.number_of_residues})"
        )


@dataclass
class Peptide(AminoacidSequence):
    """
    Represent a peptide sequence and properties derived from it.
    """

    sources: set[PeptideSource] = field(default_factory=set)

    def digest(self, enzyme: str, missed_cleavages: int = 0):
        """
        Digest a protein sequence into peptides.

        Parameters
        ----------
        enzyme : str
            Name of the proteolytic enzyme used for digestion.
        missed_cleavages : int
            Maximum number of allowed missed cleavages.

        Returns
        -------
        PeptideDigest
            Digested peptides together with the digestion parameters.
        """

        return digest_protein(
            self, enzyme=enzyme, missed_cleavages=missed_cleavages
        )

    def __repr__(self) -> str:
        return (
            f"Peptide("
            f"sequence='{self.sequence}', "
            f"sequence_length={self.number_of_residues}"
            f")"
        )


@dataclass
class PeptideDigest:
    """
    Result of enzymatic digestion of a protein sequence.

    Stores the generated peptides together with the information
    describing the digestion and, when available, the source protein.

    Parameters
    ----------
    peptides : list[oms.AASequence]
        Peptides generated by the enzymatic digestion.
    sequence : str
        Amino-acid sequence of the source protein.
    enzyme : str
        Name of the proteolytic enzyme used for digestion.
    missed_cleavages : int
        Maximum number of missed cleavages allowed during digestion.
    accession : str | None, optional
        Accession identifier of the source protein, if available.
    protein_name : str | None, optional
        Name of the source protein, if available.
    """

    peptides: list[Peptide]
    protein: Protein
    enzyme: str
    missed_cleavages: int

    # sequence: str
    # enzyme: str
    # missed_cleavages: int
    # accession: str | None = None
    # protein_name: str | None = None

    def search(self, pattern: str | None = None) -> "PeptideDigest":
        """
        Return a new PeptideDigest containing only matching peptides.

        Parameters
        ----------
        pattern : str | None
            Regular expression used to filter peptides.
            If None, all peptides are returned.

        Returns
        -------
        PeptideDigest
            New digest containing the matching peptides.
        """

        if pattern is None:
            matching_peptides = self.peptides.copy()
        else:
            regex = re.compile(pattern)

            matching_peptides = [
                peptide
                for peptide in self.peptides
                if regex.search(peptide.sequence)
            ]

        return PeptideDigest(
            peptides=matching_peptides,
            sequence=self.sequence,
            enzyme=self.enzyme,
            missed_cleavages=self.missed_cleavages,
            accession=self.accession,
            protein_name=self.protein_name,
        )


# %% Functions


def digest_protein(
    substrate: Protein,
    enzyme: str,
    missed_cleavages: int,
    accession: str | None = None,
    protein_name: str | None = None,
) -> PeptideDigest:
    """
    Digest a protein sequence into peptides.

    Parameters
    ----------
    seq : str
        Protein amino-acid sequence.
    enzyme : str
        Name of the proteolytic enzyme used for digestion.
    missed_cleavages : int
        Maximum number of allowed missed cleavages.

    Returns
    -------
    PeptideDigest
        Digested peptides together with the digestion parameters.
    """

    peptides = []
    protein = oms.AASequence.fromString(substrate.sequence)

    digestor = oms.ProteaseDigestion()
    digestor.setEnzyme(enzyme)
    digestor.setMissedCleavages(missed_cleavages)

    digestor.digest(protein, peptides)

    peptides = [Peptide(pep.toString()) for pep in peptides]

    return PeptideDigest(
        peptides=peptides,
        protein=substrate,
        enzyme=enzyme,
        missed_cleavages=missed_cleavages,
    )


def get_enzymes():
    """
    Prints out all the standard enzymes available in the OMS library.
    """
    listenzymes = []

    db = oms.ProteaseDB()
    db.getAllNames(listenzymes)

    for enzyme in listenzymes:
        enzyme = enzyme.decode("utf-8")
        print(enzyme)


def aggregate_peptides(
    digests: list[PeptideDigest],
) -> dict[str, Peptide]:
    """
    Aggregate peptides from multiple digestions.

    The peptide sequence is used as the dictionary key, while
    each value contains the sources from which that peptide originated.
    """

    peptides = {}

    for digest in digests:

        source = PeptideSource(
            protein_sequence=digest.sequence,
            enzyme=digest.enzyme,
            missed_cleavages=digest.missed_cleavages,
            accession=digest.accession,
            protein_name=digest.protein_name,
        )

        for peptide in digest.peptides:

            if peptide.sequence not in peptides:
                peptides[peptide.sequence] = peptide

            peptides[peptide.sequence].sources.add(source)

    return peptides


#  Deprecated, just for reference
def print_peptide_report(aggregated: dict[str, list[PeptideDigest]]) -> None:
    """Print an aggregated peptide report."""

    for peptide_sequence, digests in aggregated.items():
        print()

        print(f"Peptide sequence: {peptide_sequence}")

        for i, digest in enumerate(digests, start=1):

            print(f"    Source {i}:")
            print(f"        Origin: {digest.protein_name}")
            print(f"        Accession: {digest.accession}")
            print(f"        Enzyme: {digest.enzyme}")
            print(f"        Missed cleavages: " f"{digest.missed_cleavages}")


def export_as_table(
    aggregated: dict[str, Peptide], filter_rows: bool = True
) -> pd.DataFrame:
    """Create a DataFrame from aggregated peptides."""

    rows = []
    df = pd.DataFrame()

    for peptide in aggregated.values():

        for source in peptide.sources:
            rows.append(
                {
                    "peptide_sequence": peptide.sequence,
                    "source_protein": source.protein_name,
                    "protein_accession": source.accession,
                    "enzyme": source.enzyme,
                    "missed_cleavages": source.missed_cleavages,
                    "MW_Da": peptide.molecular_weight,
                    # "number_residues": peptide.number_of_residues,
                }
            )
    df = pd.DataFrame(rows)

    if filter_rows:
        df = df.sort_values(
            by=[
                "peptide_sequence",
                "protein_accession",
                "enzyme",
                "missed_cleavages",
            ],
            ascending=True,
        )

        df = df.drop_duplicates(
            subset=[
                "peptide_sequence",
                "protein_accession",
                "enzyme",
            ]
        )

    return df
