"""Manipulate Pauli strings with PauliArray."""

import warnings
from itertools import batched, product

import matplotlib.pyplot as plt
import networkx as nx
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
            self._paulis = np.array(
                np.split(self._remove_phase_pauli(paulis)[0], len(paulis)))
            phase_facts = np.array(
                np.split(self._remove_phase_pauli(paulis)[1], len(paulis)))

            if coeffs is None:
                self._coeffs = phase_facts

            elif paulis.shape != coeffs.shape:
                raise ValueError(
                    "Mismatch in pauli strings and coefficients shapes.")

            else:
                self._coeffs = coeffs * phase_facts

            for i, pauliarr in enumerate(self._paulis):
                unique_paulis, cnts = np.unique(pauliarr, return_counts=True)
                mult_occurs = unique_paulis[cnts > 1]

                if len(mult_occurs) != 0:
                    warnings.warn(
                        "Redundant paulis in an array not supported for 2D case, please reshape it to 1D!")
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
                raise ValueError(
                    "Mismatch in pauli strings and coefficients shapes.")

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
                mult_ids = list(np.argwhere(self._paulis == val).flatten()
                                for val in mult_occurs)
                # indices of paulis featuring only one time.
                only_ids = list(np.argwhere(self._paulis == val).flatten()
                                for val in only_occurs)

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

        pauli, coeffs = np.zeros(len(flat_pauli), dtype=object), np.zeros(
            len(flat_pauli), dtype=complex)

        for i in range(len(pauli)):
            if flat_pauli[i][0] == "-":
                if flat_pauli[i][1] == "i":
                    pauli[i] = flat_pauli[i].replace("-i", "")
                    coeffs[i] = 0.0 - 1j

                elif flat_pauli[i][1] in PauliArr.pauli_arr:
                    pauli[i] = flat_pauli[i].replace("-", "")
                    coeffs[i] = -1.0 + 0j
                else:
                    raise ValueError(f"Not a valid Pauli {
                                     flat_pauli[i]} string.")

            elif flat_pauli[i][0] == "i":
                if flat_pauli[i][1] in PauliArr.pauli_arr:
                    pauli[i] = flat_pauli[i].replace("i", "")
                    coeffs[i] = 0.0 + 1j
                else:
                    raise ValueError(f"Not a valid Pauli {
                                     flat_pauli[i]} string.")

            elif flat_pauli[i][0] in PauliArr.pauli_arr:
                if flat_pauli[i][1] in PauliArr.pauli_arr:
                    pauli[i] = flat_pauli[i]
                    coeffs[i] = 1.0 + 0j
                else:
                    raise ValueError(f"Not a valid Pauli {
                                     flat_pauli[i]} string.")
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
                    strout += f"Pauli String:  {pauli} Coefficient {
                        coeff.real:.2f}  {coeff.imag:.2f}j\n"

                strout += "\n"

            return strout

        else:
            strout = ""
            for pauli, coeff in zip(self._paulis, self._coeffs):
                strout += f"Pauli String:  {pauli} Coefficient {
                    coeff.real:.2f}  {coeff.imag:.2f}j\n"

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
        len_pauli_chk = np.array([len(pauli)
                                 for pauli in self._paulis.reshape(-1)])

        if np.unique(len_pauli_chk).size != 1:
            raise ValueError("All pauli strings do not have same length")
        else:
            return len_pauli_chk[0]

    @property
    def _loc_I(self) -> np.ndarray:
        """Compute the indices where pauli operator is `I`.

        Qubit indices corresponding to `I` is extracted leveraging
        x-z representation which naturally takes care of `endianness`.
        Qubit locations where both x- and z- are `0` corresponds to `I`.


        :return: A numpy array (multidim) of integers denoting indices,
               (row, column) for 2D array.
        """
        # x-z representation. ``endianness`` is taken care of in x-z representation.
        x_block = self._get_x_block
        z_block = self._get_z_block

        id_I = np.argwhere(np.bitwise_and(x_block == 0, z_block == 0))

        return id_I

    @property
    def _loc_X(self) -> np.ndarray[int]:
        """Compute the indices where pauli operator is `X`.

        Qubit indices corresponding to `X` is extracted leveraging
        x-z representation which naturally takes care of `endianness`.
        Qubit locations where x- is `1` and z- is `0` corresponds to `X`.


        :return: A numpy array (multidim) of integers denoting indices,
               (row, column) for 2D array.
        """
        # x-z representation. ``endianness`` is taken care of in x-z representation.
        x_block = self._get_x_block
        z_block = self._get_z_block

        id_X = np.argwhere(np.bitwise_and(x_block == 1, z_block == 0))

        return id_X

    @property
    def _loc_Y(self) -> np.ndarray[int]:
        """Compute the indices where pauli operator is `Y`.

        Qubit indices corresponding to `X` is extracted leveraging
        x-z representation which naturally takes care of `endianness`.
        Qubit locations where both x- and z- are `1` corresponds to `Y`.


        :return: A numpy array (multidim) of integers denoting indices,
               (row, column) for 2D array.
        """
        # x-z representation. ``endianness`` is taken care of in x-z representation.
        x_block = self._get_x_block
        z_block = self._get_z_block

        id_Y = np.argwhere(np.bitwise_and(x_block == 1, z_block == 1))

        return id_Y

    @property
    def _loc_Z(self) -> np.ndarray[int]:
        """Compute the indices where pauli operator is `Z`.

        Qubit indices corresponding to `X` is extracted leveraging
        x-z representation which naturally takes care of `endianness`.
        Qubit locations where x- is `0` and z- is `1` corresponds to `Z`.


        :return: A numpy array (multidim) of integers denoting indices,
               (row, column) for 2D array.
        """
        # x-z representation. ``endianness`` is taken care of in x-z representation.
        x_block = self._get_x_block
        z_block = self._get_z_block

        id_Z = np.argwhere(np.bitwise_and(x_block == 0, z_block == 1))

        return id_Z

    def _all_ids(self) -> np.ndarray[int]:
        """Compute array indices where pauli operator is identity.

        This supports both 1D and 2D array of paulis.

        :return: A numpy array of integers.

        """
        pauli_ids = np.argwhere(~np.bitwise_or(
            self._get_x_block, self._get_z_block).any(axis=-1))

        return pauli_ids

    @property
    def _if_hermitian(self) -> bool | np.array(bool):
        """Check if the operator corresponding to the pauli string is hermitian.

        The data structure is linear combination of paulis. The coefficients must be
        real for it to be hermitian.

        :return: A boolean (True/False) for 1D array or numpy array of boolean for 2D array.

        """
        if self._arrtype == "twodim":
            return np.array([np.all(np.isreal(ele)) for ele in self._coeffs])

        return np.all(np.isreal(self._coeffs))

    def _get_hermitian_conj(self) -> "PauliArr":
        """Return a hermitian conjugate of the `PauliArr` object.

        As pauli operators are hermitian, it just calculates the
        complex conjugate of the coefficients.

        :return: A `PauliArr` object.
        """
        return PauliArr(self._paulis, np.conj(self._coeffs))

    @staticmethod
    def _pauli_gr(n: int) -> np.ndarray[str]:
        """Compute all elements of N-qubit pauli operators.

        The pauli strings are allowed with phase factors.
        For N-qubit, total number of elements is 4**(N+1).

        :param n: An integer corresponds to number of qubits.
        :return: A 1D numpy array of pauli strings.

        """
        all_combs = np.array(
            [combination for combination in product(range(4), repeat=n)])

        assert len(all_combs) == 4 ** n

        # an overestimate of maximum string size.
        max_strsize = 5 + n

        str_set = np.empty(4 * len(all_combs), dtype="U" + str(max_strsize))

        comb_ctr = 0
        for i in range(0, len(str_set), 4):
            str_set[i] = "".join(PauliArr.pauli_arr[letter]
                                 for letter in all_combs[comb_ctr])
            str_set[i + 1] = PauliArr.phase_arr[0] + str_set[i]
            str_set[i + 2] = PauliArr.phase_arr[1] + str_set[i]
            str_set[i + 3] = PauliArr.phase_arr[2] + str_set[i]

            comb_ctr += 1

        return str_set

    @property
    def _as_dict(self) -> dict | np.ndarray[dict]:
        """Return dictionary representation of the paulis and the coefficients.

        Pauli strings and coefficients are represented as dictionary
        with paulis are keys and coefficients are values.

        :return: A dictionary.

        """
        paulis = self._paulis.reshape(-1)
        coeffs = self._coeffs.reshape(-1)

        paulidict = {}

        for pauli, coeff in zip(paulis, coeffs):
            paulidict[pauli] = coeff

        if self._arrtype == "twodim":
            paulidict = np.array(
                [dict(batch) for batch in batched(paulidict.items(),
                      self._size_paulis // self._size_pauli_arr)]
            )

        return paulidict

    @property
    def _get_paulis(self) -> np.ndarray[str]:
        """Return just the pauli strings (w/o phase) as 1D or 2D numpy array.

        :return: A 1D/2D numpy array of pauli strings.
        """
        return self._paulis

    @classmethod
    def _from_list(cls, paulilist: np.ndarray[str]) -> "PauliArr":
        """Generate a PauliArr from 1D or 2D list of pauli strings.

        The coefficients are not provided and taken as `None`and hence
        returned as 1.+0.j

        :param paulilist: A 1D/2D numpy array of pauli strings.
        :return: A PauliArr object.
        """
        return PauliArr(paulilist, None)

    @classmethod
    def _from_dict(cls, paulidict: dict | np.ndarray) -> "PauliArr":
        """Generate a PauliArr from a dictionary (or numpy array of dictionaries) of paulis and coefficients .

        :param paulidict: A dictionary of paulis and coefficients as keys and
                          values for 1D case.
                          A numpy array of dictionaries for 2D case.
        :return: A PauliArr object.
        """
        if isinstance(paulidict, np.ndarray):
            outer_keys = []
            outer_values = []
            for inner_dict in paulidict:
                outer_keys.append(list(inner_dict.keys()))
                outer_values.append(list(inner_dict.values()))

            return PauliArr(np.array(outer_keys), np.array(outer_values))

        elif isinstance(paulidict, dict):
            keys = np.array(list(paulidict.keys()))
            values = np.array(list(paulidict.values()))

            return PauliArr(keys, values)

    @property
    def _as_xz_repr(self) -> np.ndarray:
        """Compute x-z representation of the paulis.

        :return: A 2D numpy array (for 1D array of paulis) or 3D numpy array i.e. numpy array
                of 2D numpy arrays (for 2D array of paulis).

        """
        # Initialize symplectic matrix.
        symp_mat = np.zeros(
            (self._size_paulis, 2 * self._num_qbits), dtype=int)

        paulis = self._paulis.reshape(-1)

        if self._style == "little-endian":
            # as matrix indexing goes from left -> right.
            paulis = [ele[::-1] for ele in paulis]

        for row, pauli in enumerate(paulis):
            pauli_list = np.array(list(pauli), dtype=str)
            Xloc = (pauli_list == "X").astype(int)
            Yloc = (pauli_list == "Y").astype(int)
            Zloc = (pauli_list == "Z").astype(int)

            symp_mat[row][: self._num_qbits] += Xloc
            symp_mat[row][self._num_qbits:] += Zloc
            symp_mat[row][: self._num_qbits] += Yloc
            symp_mat[row][self._num_qbits:] += Yloc

        if self._arrtype == "twodim":
            symp_mat = np.array(np.split(symp_mat, self._size_pauli_arr))

        return symp_mat

    @property
    def _as_zx_repr(self) -> np.ndarray:
        """Compute z-x representation of the paulis.

        Flips the x-z matrix to z-x representation.

        :return: A 2D numpy array (for 1D array of paulis) or 3D numpy array i.e. numpy array
                    of 2D numpy arrays (for 2D array of paulis).

        """
        # Numpy array for x-z representation.
        xz_repr = self._as_xz_repr

        for i in range(self._num_qbits):
            if self._arrtype == "twodim":
                xz_repr[:, :, [i, (self._num_qbits + i)]
                                   ] = xz_repr[:, :, [(self._num_qbits + i), i]]

            else:
                xz_repr[:, [i, (self._num_qbits + i)]
                                ] = xz_repr[:, [(self._num_qbits + i), i]]

        return xz_repr

    @property
    def _get_x_block(self) -> np.ndarray:
        """Return X-block of the x-z representation of the paulis.

        :return: A 2D numpy array (for 1D array of paulis) or 3D numpy array i.e. numpy array
                of 2D numpy arrays (for 2D array of paulis).
        """
        if self._arrtype == "twodim":
            return self._as_xz_repr[:, :, : self._num_qbits]

        return self._as_xz_repr[:, : self._num_qbits]

    @property
    def _get_z_block(self) -> np.ndarray:
        """Return Z-block of the x-z representation of the paulis.

        :return: A 2D numpy array (for 1D array of paulis) or 3D numpy array i.e. numpy array
                of 2D numpy arrays (for 2D array of paulis).
        """
        if self._arrtype == "twodim":
            return self._as_xz_repr[:, :, self._num_qbits:]

        return self._as_xz_repr[:, self._num_qbits:]

    @staticmethod
    def _merge_xz(x_block: np.ndarray, z_block: np.ndarray) -> np.ndarray:
        """Merge x- and z- blocks to create x-z representation.

        :return: A numpy array corresponding to x-z representation of the paulis.
        """
        if all(isinstance(val, str) for val in [paulis[0][0], coeffs[0][0]]):
            return np.concatenate((x_block, z_block), axis=1)

        elif all(isinstance(val, np.ndarray) for val in [paulis[0][0], coeffs[0][0]]):
            return np.concatenate((x_block, z_block), axis=2)

    @staticmethod
    def _check_commute_paulis(pauli_1: str, pauli_2: str, qwc: bool = False) -> bool:
        """Check if two Pauli strings are mutually commuting :footcite:`gokhale2019minimizing`.

        It can execute both qubitwise commuting (qwc) and general commuting (gc) based on user
        input.

        References
        ==========
        .. footbibliography::

        :raises ValueError: If Pauli strings have different lengths.
        :param pauli_1: A string denoting the first pauli string.
        :param pauli_2: A string denoting the second pauli string.
        :param qwc: A boolean. (True / False) if checking is done based on (qwc / gc).
        :return: A boolean. (True / False) if the two pauli strings are mutually commuting or not.

        """
        if len(pauli_1) != len(pauli_2):
            raise ValueError("Length of the two strings must be equal!")

	    # counter for non-commuting cases for qubit-wise commuting.
        k_count = 0
        for i, j in zip(pauli_1, pauli_2):
            if "I" not in (i, j):
                if i != j:
                    k_count += 1

        if qwc:
            return True if k_count == 0 else False
        else:
            return True if k_count % 2 == 0 else False

    def _get_pairwise_compatibility(self, rule: str = 'qwc') -> np.ndarray:
        """Compute pairwise compatibility of a set of paulis based on rule.

        :param rule:
        :return: A numpy array of tuples denoting the adjacency matrix of the
                graph.
        """
        assert rule in {'qwc', 'gc', 'ac'}, "Compatibility rule not supported!"

        edge_arr = []

        for i in range(len(self._paulis)):
            for j in range(i+1, len(self._paulis)):
                if _check_commute_paulis(self._paulis[i], self._paulis[j]) and rule in ("qwc", "gc"):
                    edge_arr.append((i, j))
                elif rule == "ac":
                    edge_arr.append((i, j))

        return np.array(edge_arr)

    def _get_compatibility_graph(self, rule: str = 'qwc', visualize: bool = False) -> np.ndarray:
        """Generate compatibility graph of pauli strings based on rule.

        :param rule: A string denoting compatibility rule of the graph.
        :return:  A numpy array corresponding to the adjacency matrix.
        """
        # An empty networkx graph.
        G = nx.Graph()
        # adding the paulis as nodes of the Graph.
        G.add_nodes_from(self._paulis)

        # create graph edges based on compatibility rule.
        assert rule in {'qwc', 'gc', 'ac'}, "Compatibility rule not supported!"

        # create edges of the graph from the compatibility rule.
        match rule:
            case "qwc":
                compatibility_arr = self._get_pairwise_compatibility(
                    rule="qwc")
                G.add_edges_from(compatibility_arr)
            case "gc":
                compatibility_arr = self._get_pairwise_compatibility(rule="gc")
                G.add_edges_from(compatibility_arr)
            case "ac":
                compatibility_arr = self._get_pairwise_compatibility(rule="ac")
                G.add_edges_from(compatibility_arr)

        if visualize:
            nx.draw(G, with_labels=True, node_color='cyanblue',
                    edge_color='black', node_size=1500)
            plt.show()

        # adjacency matrix in numpy array form.
        adjmat = nx.to_numpy_array(nx.adjacency_matrix(G))

        return adjmat

    def _get_cliques(self,
        method: str = "max-clique",
        graph_type: str = "qwc",
        greedy_method: str = "largest-first",
        clique_method: str = "BK"
    ) -> "PauliArr":
        """Compute commuting or anticommuting cliques based on strategy.

        'largest_first', 'random_sequential', 'smallest_last', 'independent_set',
        'connected_sequential_bfs', 'connected_sequential_dfs', 'DSATUR'.




        :param strategy: A str denoting strategy for generating cliques.
        :return:
        """
        # Get the compatibility graph.
        G = self._get_compatibility_graph(rule=graph_type)

        assert method in {
            "max-clique", "greedy-color"}, "Partition finding method not supported!"

        if method == "max_clique":

            assert clique_method in {"BK", "BH"}, "Clique finding method not supported!"

            if clique_method == "BK":
                list(nx.find_cliques(G))

            elif clique_method == "BH":
                raise NotImplementedError("This implementation is not stable yet.")

        elif method == "greedy-color":

            assert greedy_method in {"largest-first", "random_sequential", "smallest_last", "independent_set",
                "connected_sequential_bfs", "connected_sequential_dfs", "DSATUR"}, "Greedy-coloring method not supported!"

            nx.algorithms.coloring.greedy_color(G, strategy = "largest_first")



