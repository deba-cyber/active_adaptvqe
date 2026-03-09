"""Manipulate Pauli strings with PauliArray."""


import warnings

import numpy as np


class PauliArr:
    r"""Create a PauliArr object.

    Efficient representation and manipulation of a set of pauli strings
    with convenient data structures, combining ideas introduced mainly in
    :footcite: `dion2024efficiently` and `kawase2023fast`.

    References
    ==========
    .. footbibliography::

    """

    # Binary encoding of the three pauli operators and the identity into two bits: x-z decomposition.
    pauli_xz_dict = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}

    # pauli operators as list of strings.
    pauli_arr = ["I", "X", "Y", "Z"]

    # allowed phase factors.
    phase_arr = ["-", "i", "-i"]

    # pauli operators in matrix form (numpy array).
    pauli_I = np.eye(2)
    pauli_X = np.array([[0, 1], [1, 0]])
    pauli_Y = np.array([[0, -1j], [1j, 0]])
    pauli_Z = np.array([[1, 0], [0, -1]])

    def __init__(
        self, paulis: np.ndarray[str], coeffs: np.ndarray[complex] | None, style: str = "little-endian"
    ) -> None:
        """Initialize variables of PauliArr class.

        Input paulis are allowed to have phase factors which are absorbed in complex coefficients
        in class initialization.
        For pauli strings, `little-endian` means qubit indexing is from right -> left with rightmost
        qubit having the lowest index. `big-endian` means qubit indexing is from left -> right with leftmost
        qubit having the lowest index.

        If a pauli string is repeated, it is taken care of by modifying the complex coefficient for 1D case.
        For 2D case, it is not dealt with, issues a warning, asking the array to be reshaped to 1D array.
        This is for keeping consistency with numpy array, as heterogeneous array not supported in numpy.

        :raises ValueError: If any string is NOT a valid pauli string or
                            number of coefficients is NOT equal to number of strings
                            (when `coeffs` is NOT `None`).
        :param paulis: A numpy array (1D or 2D) of strings.
        :param coeffs: A numpy array (1D or 2D) of complex coefficients. Input can also be `None`.
        :param style: A string denoting qubit ordering in a pauli string. Default is `little-endian`.

        """
        # Allowed characters in the string for a valid pauli string.
        # Allowed characters in the string[2:] for a valid pauli string.
        allowed_chars_wo_ph = "IXYZ"

        assert style in {"big-endian", "little-endian"}

        self._style = style

        # variable for 1D or 2D array.
        self._arrtype = ""

        if isinstance(paulis[0], np.ndarray):
            self._arrtype = "twodim"
            # checking if valid pauli string for all strings.
            for arr in paulis:
                for pauli in arr:
                    if not all(char in allowed_chars_wo_ph for char in pauli[2:]):
                        raise ValueError("Invalid Pauli string in array.")

            self._init_paulis = paulis
            self._init_coeffs = coeffs

            # separating phase factors if any.
            self._paulis = np.array(np.split(self._remove_phase_pauli(paulis)[0], len(paulis)))
            phase_facts = np.array(np.split(self._remove_phase_pauli(paulis)[1], len(paulis)))

            if coeffs is None:
                self._coeffs = phase_facts

            elif paulis.shape != coeffs.shape:
                raise ValueError("Mismatch in pauli strings and coefficients shapes.")

            else:
                self._coeffs = coeffs * phase_facts

            for i, pauliarr in enumerate(self._paulis):
                unique_paulis, cnts = np.unique(pauliarr, return_counts=True)
                mult_occurs = unique_paulis[cnts > 1]

                if len(mult_occurs) != 0:
                    warnings.warn("Redundant paulis in an array not supported for 2D case, please reshape it to 1D!")
                    break

        elif isinstance(paulis[0], str):
            self._arrtype = "onedim"
            for pauli in paulis:
                if not (all(char in allowed_chars_wo_ph for char in pauli[2:])):
                    raise ValueError("Invalid Pauli string in array.")

            self._init_paulis = paulis
            self._init_coeffs = coeffs

            # separating phase factors if any.
            self._paulis = self._remove_phase_pauli(paulis)[0]
            phase_facts = self._remove_phase_pauli(paulis)[1]

            if coeffs is None:
                self._coeffs = phase_facts

            elif paulis.shape != coeffs.shape:
                raise ValueError("Mismatch in pauli strings and coefficients shapes.")

            else:
                self._coeffs = coeffs * phase_facts

            # repeated occurence of a pauli is considered.
            # taking care of the repeated occurences and combining the coefficients.
            unique_paulis, cnts = np.unique(self._paulis, return_counts=True)
            mult_occurs = unique_paulis[cnts > 1]
            only_occurs = unique_paulis[cnts == 1]


            if len(mult_occurs) != 0:
                mult_occurs_dict, only_occurs_dict = {}, {}
                # indices of paulis featuring multiple times.
                mult_ids = list(np.argwhere(self._paulis == val).flatten() for val in mult_occurs)
                # indices of paulis featuring only one time.
                only_ids = list(np.argwhere(self._paulis == val).flatten() for val in only_occurs)

                for x, y in zip(mult_occurs, mult_ids):
                    mult_occurs_dict[x] = sum(self._coeffs[y])

                for x, y in zip(only_occurs, only_ids):
                    only_occurs_dict[x] = self._coeffs[y]

                self._coeffs = np.zeros(len(unique_paulis), dtype=complex)
                self._paulis = unique_paulis

                for i, pauli in enumerate(unique_paulis):
                    if pauli in mult_occurs:
                        self._coeffs[i] = mult_occurs_dict[pauli]

                    else:
                        self._coeffs[i] = only_occurs_dict[pauli]


    def _remove_phase_pauli(self, init_paulis: np.ndarray[str]) -> tuple[np.ndarray[str], np.ndarray[complex]]:
        r"""Remove phase factors i.e. `-`,`i`, `-i` from a list of pauli strings.

        Regardless of 1D or 2D array of pauli strings originally, the array is flattened
        before processing for discarding phase factors. The phase factors are returned
        as numpy array of complex numbers.

        :raises ValueError: If second character of the string is either `-` or `i` when first
                        character is `i`.
        :param init_paulis: A numpy array (1D or 2D) of pauli strings.
        :return: A tuple of 1D numpy arrays of [strings (Pauli strings), complex numbers
                    (coefficients)].

        """
        flat_pauli = init_paulis.reshape(-1)

        pauli, coeffs = np.zeros(len(flat_pauli), dtype=object), np.zeros(len(flat_pauli), dtype=complex)

        for i in range(len(pauli)):
            if flat_pauli[i][0] == "-":
                if flat_pauli[i][1] == "i":
                    pauli[i] = flat_pauli[i].replace("-i", "")
                    coeffs[i] = 0.0 - 1j

                elif flat_pauli[i][1] in PauliArr.pauli_arr:
                    pauli[i] = flat_pauli[i].replace("-", "")
                    coeffs[i] = -1.0 + 0j
                else:
                    raise ValueError(f"Not a valid Pauli {flat_pauli[i]} string.")

            elif flat_pauli[i][0] == "i":
                if flat_pauli[i][1] in PauliArr.pauli_arr:
                    pauli[i] = flat_pauli[i].replace("i", "")
                    coeffs[i] = 0.0 + 1j
                else:
                    raise ValueError(f"Not a valid Pauli {flat_pauli[i]} string.")

            elif flat_pauli[i][0] in PauliArr.pauli_arr:
                if flat_pauli[i][1] in PauliArr.pauli_arr:
                    pauli[i] = flat_pauli[i]
                    coeffs[i] = 1.0 + 0j
                else:
                    raise ValueError(f"Not a valid Pauli {flat_pauli[i]} string.")
            else:
                raise ValueError(f"Not a valid Pauli {flat_pauli[i]} string.")

        return (np.array(pauli), np.array(coeffs))

    def __repr__(self) -> str:
        flat_pauli = self._paulis.reshape(-1)
        flat_coeffs = self._coeffs.reshape(-1)
        return f"PauliArr object(Pauli Strings={flat_pauli}, Coefficients={flat_coeffs})"


    def __str__(self) -> str:
        """String representation of the object in user-friendly readable format."""
        if self._arrtype == "twodim":
            strout = ""
            for i in range(self._size_pauli_arr):
                for pauli, coeff in zip(self._paulis[i], self._coeffs[i]):
                    strout += f"Pauli String:  {pauli} Coefficient {coeff.real:.2f} + {coeff.imag:.2f}j\n"

                strout += "\n"

            return strout

        else:
            strout = ""
            for pauli, coeff in zip(self._paulis, self._coeffs):
                strout += f"Pauli String:  {pauli} Coefficient {coeff.real:.2f} + {coeff.imag:.2f}j\n"

            return strout

    @property
    def _size_paulis(self) -> int:
        """Compute total number of pauli strings regardless of the array dimension.

        :return: An integer denoting total number pauli strings in the 1D or 2D array of paulis.

        """
        return self._paulis.size

    @property
    def _size_pauli_arr(self) -> int:
        """Compute total number of arrays in paulis.

        :return: An integer denoting number of arrays. For one-dimension, returns 1.

        """
        if self._arrtype == "onedim":
            return 1

        return len(self._paulis)

    @property
    def _shape_pauli_arr(self) -> tuple[int, ...]:
        """Compute shape of pauli array."""
        return self._paulis.shape

    @property
    def _num_qbits(self) -> int:
        """Compute the number of qubits in the paulis.

        :raises ValueError: If all pauli strings are not of same length.
        :return: An integer denoting number of qubits in the pauli.

        """
        len_pauli_chk = np.array([len(pauli) for pauli in self._paulis.reshape(-1)])

        if np.unique(len_pauli_chk).size != 1:
            raise ValueError("All pauli strings do not have same length")
        else:
            return len_pauli_chk[0]

