"""
LayerCode.py

Supporting code for the paper:

    Layer codes as partially self-correcting quantum memories
    Shouzhen Gu*, Libor Caha*, Shin Ho Choe, Zhiyang He, Aleksander Kubica, and Eugene Tang
    arXiv:2510.06659. *These authors contributed equally to this work.

This module constructs the layer codes of Williamson and Baspin from input
CSS parity-check matrices and generates the resulting X- and Z-type stabiliser generators.

If you use this code, please cite the paper above.
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from math import floor

class LayerCode:

    def __init__(self, input_xChecks, input_zChecks, c_x=2, c_y=2, c_z=2):
        # parity check matrices defining the layer code
        self.input_xChecks = np.asarray(input_xChecks, dtype=int)
        self.input_zChecks = np.asarray(input_zChecks, dtype=int)

        # number of layers of each type
        self.num_qLayer = self.input_xChecks.shape[1]
        self.num_xLayer = self.input_xChecks.shape[0]
        self.num_zLayer = self.input_zChecks.shape[0]
        self.num_layer = self.num_qLayer + self.num_xLayer + self.num_zLayer

        # The parameters c_x, c_y, and c_z serves as the counterpart to the superlattice spacing constant "c" used in the paper.
        # I allow the possibility that the constant is different in each dimension.
        # In the current version, c = 1 is allowed.
        self.c_x = c_x
        self.c_y = c_y
        self.c_z = c_z

        self.x_max = (self.num_zLayer+1)*self.c_x
        self.z_max = (self.num_xLayer+1)*self.c_z
        self.y_max = self.num_qLayer*self.c_y

        # number of qubits in each Q layer
        self.num_qubits_qLayer = 2 * self.x_max * self.z_max + 3 * self.x_max + self.z_max + 2

        # the qubit_bins array stores the linear coordinate of the first qubit in each layer
        self.qubit_bins = []
        self.num_total_qubits = 0

        for i in range(self.num_qLayer):
            self.qubit_bins.append(self.num_total_qubits)
            self.num_total_qubits += self.num_qubits_qLayer

        for i in range(self.num_xLayer):
            self.qubit_bins.append(self.num_total_qubits)
            self.num_total_qubits += self.num_qubit_xLayer(i)

        for i in range(self.num_zLayer):
            self.qubit_bins.append(self.num_total_qubits)
            self.num_total_qubits += self.num_qubit_zLayer(i)

        self.x_defects, self.y_defects, self.z_defects = self.build_defects()

        # stores the positions of the y-defects on each X (resp. Z) layer.
        # each X-layer is partitioned into strips, where each strip is the region between two qubits in its support
        # y_defect_on_X[i][j] is then a list of x coordinates of the non-trivial y-defects on the ith X-layer within
        # the jth strip (analogous for Z)

        self.y_defect_on_X = []
        self.y_defect_on_Z = []

        for i in range(self.num_xLayer):
            self.y_defect_on_X.append([])
            support = self.support_xLayer(i)
            for q in range(len(support)-1):
                self.y_defect_on_X[i].append([])
                for j in range(self.num_zLayer):
                    if support[q] in self.y_defects[i][j] and self.y_defects[i][j][support[q]] != 2:
                        self.y_defect_on_X[i][q].append((j+1)*self.c_x)

        for j in range(self.num_zLayer):
            self.y_defect_on_Z.append([])
            support = self.support_zLayer(j)
            for q in range(len(support)-1):
                self.y_defect_on_Z[j].append([])
                for i in range(self.num_xLayer):
                    if support[q] in self.y_defects[i][j] and self.y_defects[i][j][support[q]] != 2:
                        self.y_defect_on_Z[j][q].append((i+1)*self.c_z)

        # stores the region_id of the first region on each layer
        self.region_bins = []
        self.num_regions = 0

        for i in range(self.num_qLayer):
            self.region_bins.append(self.num_regions)
            self.num_regions += (len(self.x_defects[i])+1)*(len(self.z_defects[i])+1)

        for i in range(self.num_xLayer):
            self.region_bins.append(self.num_regions)
            support = self.support_xLayer(i)
            for q in range(len(support)-1):
                self.num_regions += len(self.y_defect_on_X[i][q])+1

        for i in range(self.num_zLayer):
            self.region_bins.append(self.num_regions)
            support = self.support_zLayer(i)
            for q in range(len(support) - 1):
                self.num_regions += len(self.y_defect_on_Z[i][q]) + 1

        # list of check operators for the current layer code
        # each element of the list is a tuple containing the linear coordinates of the qubits belonging to that check
        self.xCheck_list = self.build_XStabilizers()
        self.zCheck_list = self.build_ZStabilizers()



    # returns the parity check matrices of a pair random binary codes
    # satisfying the CSS condition
    # n is the total number of bits/qubits
    # m_x and m_z are the total number of (independent) x and z type checks, respectively
    # data type returned is a GF(2) matrix
    @staticmethod
    def generate_random_CSS(n, m_x, m_z, rng=None):
        assert m_x+m_z < n, "n - m_x - m_z must be less than 0."
        import galois as gal

        if rng is None:
            rng = np.random.default_rng()

        # check of random code with dimension k
        gf2 = gal.GF2

        # random sample a code C1, together with a subspace of C1
        # the subspace will be C2^*
        # define l+k as the dimension of the first code C1
        # l as the dimension of the dual of the second code C2^*
        # C2 itself has dimension n-l
        l = m_z
        k = n-l-m_x

        while True:
            # generate random matrices until we get one with full rank
            while True:
                gen2 = gf2(rng.integers(2, size=(n-l, n)))
                # gen2 = gf2.Random((n - l, n))
                if np.linalg.matrix_rank(gen2) == n - l:
                    break

            rref_gen2 = gen2.row_reduce(eye="left")

            i = 0
            for i in range(0, n - l):
                for j in range(i, n):
                    if rref_gen2[i, j] == 1:
                        rref_gen2[:, [i, j]] = rref_gen2[:, [j, i]]
                        gen2[:, [i, j]] = gen2[:, [j, i]]
                        break

            check1 = gen2[0:n - k - l, :]
            a = rref_gen2[:, n - l:].transpose()
            check2 = np.concatenate((a, gf2(np.identity(l, dtype=int))), axis=1)

            # this part is somewhat awkward, but for small parameters
            # it is common to have a check matrix with a row that has only a single non-zero entry
            # this not only corresponds to a distance 0 CSS code, but is also invalid
            # as an input to the layer code construction
            # for now, we repeatedly sample until all rows have at least two non-zero elements
            if all(np.count_nonzero(row) > 1 for row in check1)\
                    and all(np.count_nonzero(row) > 1 for row in check2):
                break

        return check1, check2

########################################################################################################################

    # builds the defect structure of the layer code
    # defects are stored in 3 lists x_defects, y_defects, and z_defects,
    # each parameterizing the defects of the respective type
    # x_defects[i] stores the list of z-coordinates of the defects on qubit layer i
    # likewise z_defects[i] stores the list of x-coordinates of the defects on qubit layer i

    # y_defects is a num_xLayer x num_zLayer matrix. The (i,j)th entry stores a dictionary of pairs {qlayer:junction_type}
    # qlayer indicates where the ith X-layer and the jth Z-layer has non-trivial y-junction
    # junction_type indicates the type of junction, with 0 being a starting y-junction defect (i.e., non-trivial on top),
    # 1 being intermediate (i.e., non-trivial on both sides), and 2 being an ending defect (i.e., non-trivial on bottom)
    def build_defects(self):
        x_defects = []
        y_defects = []
        z_defects = []

        # build x_defects and y_defects
        for layer in range(self.num_qLayer):
            row = []
            # x defects
            for i in range(self.num_xLayer):
                if self.input_xChecks[i][layer] == 1:
                    row.append((i + 1) * self.c_z)
            x_defects.append(row)

            row = []
            # z defects
            for j in range(self.num_zLayer):
                if self.input_zChecks[j][layer] == 1:
                    row.append((j + 1) * self.c_x)
            z_defects.append(row)

        # y_defects
        for i in range(self.num_xLayer):
            y_defects.append([])
            for j in range(self.num_zLayer):
                y_defects[i].append({})
                overlap = self.input_xChecks[i] + self.input_zChecks[j]
                is_trivial = True

                for k in range(len(overlap)):
                    if overlap[k] == 2 and is_trivial == True:
                        is_trivial = False
                        y_defects[i][j].update({k: 0})
                    elif overlap[k] != 2 and is_trivial == False:
                        y_defects[i][j].update({k: 1})
                    elif overlap[k] == 2 and is_trivial == False:
                        is_trivial = True
                        y_defects[i][j].update({k: 2})

        return x_defects, y_defects, z_defects

    def support_xLayer(self, n):
        return np.nonzero(self.input_xChecks[n])[0]

    def support_zLayer(self, n):
        return np.nonzero(self.input_zChecks[n])[0]

    # returns the index of the first and last qubit of the nth X-layer
    def span_xLayer(self, n):
        support = np.nonzero(self.input_xChecks[n])[0]
        return support[0], support[-1]

    # returns the index of the first and last qubit of the nth X-layer
    def span_zLayer(self, n):
        support = np.nonzero(self.input_zChecks[n])[0]
        return support[0], support[-1]

    # returns the standard surface code star operator at location (x,y,z) on Q (layer_type=0), X (layer_type=1)
    # or Z (layer_type=2) layers, respectively.
    def get_star(self, x, y, z, layer_type):
        s = []

        if layer_type == 0:
            # black star
            s.append(self.lattice_to_linear(x, y, z, 2, 0))
            s.append(self.lattice_to_linear(x, y, z - 1, 2, 0))
            s.append(self.lattice_to_linear(x, y, z, 0, 0))
            s.append(self.lattice_to_linear(x - 1, y, z, 0, 0))

        elif layer_type == 1:
            # blue star
            s.append(self.lattice_to_linear(x, y, z, 0, 1))
            s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
            s.append(self.lattice_to_linear(x, y, z, 1, 1))
            s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))

        elif layer_type == 2:
            # red star
            s.append(self.lattice_to_linear(x, y, z, 2, 2))
            s.append(self.lattice_to_linear(x, y, z - 1, 2, 2))
            s.append(self.lattice_to_linear(x, y, z, 1, 2))
            s.append(self.lattice_to_linear(x, y - 1, z, 1, 2))

        return s

    # returns the standard surface code plaquette operator at location (x,y,z) on Q (layer_type=0), X (layer_type=1)
    # or Z (layer_type=2) layers, respectively. The plaquette at (x,y,z) is defined to be the square with (x,y,z) as the
    # vertex nearest to the origin
    def get_square(self, x, y, z, layer_type):
        s = []

        if layer_type == 0:
            # black square
            s.append(self.lattice_to_linear(x, y, z, 2, 0))
            s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
            s.append(self.lattice_to_linear(x, y, z, 0, 0))
            s.append(self.lattice_to_linear(x, y, z + 1, 0, 0))

        elif layer_type == 1:
            # blue square
            s.append(self.lattice_to_linear(x, y, z, 0, 1))
            s.append(self.lattice_to_linear(x, y + 1, z, 0, 1))
            s.append(self.lattice_to_linear(x, y, z, 1, 1))
            s.append(self.lattice_to_linear(x + 1, y, z, 1, 1))

        elif layer_type == 2:
            # red square
            s.append(self.lattice_to_linear(x, y, z, 2, 2))
            s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
            s.append(self.lattice_to_linear(x, y, z, 1, 2))
            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))

        return s

    # returns the number of qubits in the nth X-layer
    def num_qubit_xLayer(self, n):
        first, last = self.span_xLayer(n)
        return 2 * self.x_max * (last - first) * self.c_y + (last - first) * self.c_y + self.x_max

    # returns the number of qubits in the nth Z-layer
    def num_qubit_zLayer(self, n):
        first, last = self.span_zLayer(n)
        return 2 * self.z_max * ((last - first) * self.c_y - 2) + ((last - first) * self.c_y - 2) + \
            self.z_max + 2 * (self.z_max + 1) + 2 * ((last - first) * self.c_y - 1)

    # returns the layer_type corresponding to a given layer_number
    # 0 for Q, 1 for X and 2 for Z
    def get_layer_type(self, layer_number):
        if layer_number < self.num_qLayer:
            layer_type = 0
        elif layer_number < (self.num_qLayer + self.num_xLayer):
            layer_type = 1
        else:
            layer_type = 2

        return layer_type

    # For convenience, we will define three types of coordinates on the qubits of the layer code
    # the first type of coordinate system will be the "lattice coordinates" of the qubit
    # lattice coordinates are a 5-tuple of the form (x,y,z,e, l), where (x,y,z) denotes the (global) coordinates of
    # a vertex and e (from 0 to 2) indicates the edge direction from that vertex, with 0, 1, 2 corresponding to
    # x, y, z, respectively. For example (4,2,6,0) would indicate a qubit located on the edge from vertices
    # (4,2,6) to (5,2,6). The final label l is the layer type of the edge, indicating which type of layer (Q,X,Z) = (0,1,2)
    # it lives on. This last label is only needed to break degeneracies where parallel edges are present when planes intersect.

    # We will also define a "linear coordinate", which is just a global ordering on the set of all qubits.
    # By convention, all qubits will be ordered by layer, with Q-type layers coming first, then X-type layers, then Z-type layers.
    # See figure for ordering within each layer

    # Finally, we define a "local coordinate", which is a pair (layer_num, local_coord). The layer number indicates
    # which layer the qubit lives on, while the local_coordinate is a linear ordering on the qubits of that layer, again
    # ordered according to the relevant figures.

    # given the linear coordinate number of a qubit, return its lattice coordinates
    def linear_to_lattice(self, lin_coord):
        layer_num, local_coord = self.linear_to_local(lin_coord)
        return self.local_to_lattice(layer_num, local_coord)

    # given the linear coordinate number of a qubit, return its local coordinates
    def linear_to_local(self, lin_coord):
        layer_num = 0
        for i in range(self.num_layer):
            if self.qubit_bins[i] <= lin_coord:
                layer_num = i

        local_coord = lin_coord - self.qubit_bins[layer_num]
        return layer_num, local_coord

    def get_endpoints(self, qubit_num):
        x, y, z, edge_type, layer_type = self.linear_to_lattice(qubit_num)
        endpoint1 = (x, y, z)
        if edge_type == 0:
            endpoint2 = (x + 1, y, z)
        elif edge_type == 1:
            endpoint2 = (x, y + 1, z)
        else:
            endpoint2 = (x, y, z + 1)

        return endpoint1, endpoint2

    # given the layer number and the coordinate number of a qubit on that layer, return the global coordinates of the qubit
    def local_to_lattice(self, layer_number, local_coord):
        layer_type = self.get_layer_type(layer_number)

        if layer_type == 0:
            y = layer_number * self.c_y
            modulus = 2 * self.x_max + 1
            z = floor(local_coord / modulus) - 1
            x = local_coord % modulus
            edge_type = 2

            if x > self.x_max:
                x = x - (self.x_max + 1)
                z += 1
                edge_type = 0

            return x, y, z, edge_type, layer_type

        elif layer_type == 1:
            z = (layer_number - self.num_qLayer + 1) * self.c_z
            modulus = 2 * self.x_max + 1
            y = floor(local_coord / modulus)
            y += self.span_xLayer(layer_number - self.num_qLayer)[0] * self.c_y
            x = local_coord % modulus
            edge_type = 0
            if x >= self.x_max:
                x = x - self.x_max
                edge_type = 1

            return x, y, z, edge_type, layer_type

        else:
            x = (layer_number - self.num_qLayer - self.num_xLayer + 1) * self.c_x
            modulus = 2 * self.z_max + 3
            y = floor(local_coord / modulus)
            y += self.span_zLayer(layer_number - self.num_qLayer - self.num_xLayer)[0] * self.c_y
            z = local_coord % modulus
            edge_type = 1

            if z > self.z_max:
                y += 1
                z = z - (self.z_max + 2)
                edge_type = 2

            return x, y, z, edge_type, layer_type

    # local coordinates to linear coordinates
    def local_to_linear(self, layer_num, local_coord, verbose=False):
        if verbose:
            print(layer_num, local_coord)

        return self.qubit_bins[layer_num] + local_coord

    # given lattice coordinates of a qubit, return its local coordinates
    def lattice_to_local(self, x, y, z, edge_type, layer_type, verbose=False):
        if layer_type == 0:
            layer_number = floor(y / self.c_y)
            modulus = 2 * self.x_max + 1

            if edge_type == 2:
                z += 1

            local_coord = z * modulus

            if edge_type == 2:
                local_coord += x
            else:
                local_coord += self.x_max + 1 + x

            return layer_number, local_coord

        elif layer_type == 1:
            layer_number = self.num_qLayer + floor(z / self.c_z) - 1
            y -= self.span_xLayer(floor(z / self.c_z) - 1)[0] * self.c_y
            modulus = 2 * self.x_max + 1

            local_coord = y * modulus

            if edge_type == 0:
                local_coord += x
            else:
                local_coord += self.x_max + x

            return layer_number, local_coord

        else:
            layer_number = self.num_qLayer + self.num_xLayer + floor(x / self.c_x) - 1
            y -= self.span_zLayer(floor(x / self.c_x) - 1)[0] * self.c_y
            modulus = 2 * self.z_max + 3

            if verbose:
                print(layer_number)
                print(y)
                print(modulus)

            if edge_type == 2:
                y -= 1

            local_coord = y * modulus

            if edge_type == 1:
                local_coord += z
            else:
                local_coord += self.z_max + 2 + z

            return layer_number, local_coord

    # given lattice coordinates, returns the linear coordinate
    def lattice_to_linear(self, x, y, z, edge_type, layer_type, verbose=False):
        layer_number, local_coord = self.lattice_to_local(x, y, z, edge_type, layer_type, verbose)
        return self.local_to_linear(layer_number, local_coord, verbose)

    # returns the y coordinate of the nth qubit layer
    def coord_q_layer(self, n):
        return n * self.c_y

    # returns the z coordinate of the nth X-check layer
    def coord_x_layer(self, n):
        return (n + 1) * self.c_z

    # returns the x coordinate of the nth Z-check layer
    def coord_z_layer(self, n):
        return (n + 1) * self.c_x

    # returns the set of X type stabilizers as an m x num_qubit matrix
    def build_XStabilizers(self):
        stabilizers = []

        # first we build stabilizers on Q layers
        for layer in range(self.num_qLayer):
            y = layer * self.c_y

            # these dictionaries define the positions of the junctions (incidences) in the current Q layer
            # each element of the list is a pair, containing an x (or z) coordinate referencing the position
            # of the junction, as well as a index 0, 1, or 2, referencing the type of incidence.
            # index 0 denotes that the Q layer serves as a starting point, 1 denotes that the layer is intermediate
            # and 2 denotes an endpoint
            xjunctions = {}
            zjunctions = {}

            for i in range(self.num_xLayer):
                minq, maxq = self.span_xLayer(i)
                if minq <= layer <= maxq:
                    if layer == minq:
                        xjunctions.update({(i + 1) * self.c_z: 0})
                    elif layer == maxq:
                        xjunctions.update({(i + 1) * self.c_z: 2})
                    else:
                        xjunctions.update({(i + 1) * self.c_z: 1})

            for i in range(self.num_zLayer):
                minq, maxq = self.span_zLayer(i)
                if minq <= layer <= maxq:
                    if layer == minq:
                        zjunctions.update({(i + 1) * self.c_x: 0})
                    elif layer == maxq:
                        zjunctions.update({(i + 1) * self.c_x: 2})
                    else:
                        zjunctions.update({(i + 1) * self.c_x: 1})

            for x in range(self.x_max + 1):
                for z in range(self.z_max + 1):
                    # point on left boundary
                    if x == 0:
                        # normal black star present on all boundary points
                        s = []
                        s.append(self.lattice_to_linear(x, y, z - 1, 2, 0))
                        s.append(self.lattice_to_linear(x, y, z, 0, 0))
                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                        stabilizers.append(s)

                        # point defect on boundary
                        if z in xjunctions:
                            x_intersects = self.input_xChecks[floor(z / self.c_z) - 1][layer]
                            x_incidence_type = xjunctions[z]

                            # type 7 boundary point defect
                            if x_incidence_type == 0:
                                # modified blue star
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                stabilizers.append(s)

                            # type 9 boundary point defect
                            elif x_incidence_type == 2:
                                # modified blue star
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                stabilizers.append(s)

                            else:
                                # type 11 boundary defect
                                if x_intersects == 1:
                                    # modified blue star
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    stabilizers.append(s)

                                # usual (boundary) stabilizer on X-layer
                                else:
                                    # normal blue star
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    stabilizers.append(s)

                    # point on right boundary
                    elif x == self.x_max:
                        # normal black star present on all boundary points
                        s = []
                        s.append(self.lattice_to_linear(x, y, z - 1, 2, 0))
                        s.append(self.lattice_to_linear(x - 1, y, z, 0, 0))
                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                        stabilizers.append(s)

                        # point defect on boundary
                        if z in xjunctions:
                            x_intersects = self.input_xChecks[floor(z / self.c_z) - 1][layer]
                            x_incidence_type = xjunctions[z]

                            # type 8 boundary point defect
                            if x_incidence_type == 0:
                                # modified blue star
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                stabilizers.append(s)

                            # type 10 boundary point defect
                            elif x_incidence_type == 2:
                                # modified blue star
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                stabilizers.append(s)

                            else:
                                # type 12 boundary defect
                                if x_intersects == 1:
                                    # modified blue star
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                    s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    stabilizers.append(s)

                                # usual boundary defects
                                else:
                                    # normal blue star
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                    s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                    stabilizers.append(s)

                    # point in a Z junction
                    elif x in zjunctions:
                        z_intersects = self.input_zChecks[floor(x / self.c_x) - 1][layer]

                        # check if it is a point defect
                        if z in xjunctions:
                            x_intersects = self.input_xChecks[floor(z / self.c_z) - 1][layer]

                            x_incidence_type = xjunctions[z]
                            z_incidence_type = zjunctions[x]

                            # type 1 bulk point defect
                            if x_incidence_type == 0 and z_incidence_type == 0:
                                # modified blue star
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                stabilizers.append(s)

                                # modified black star
                                s = self.get_star(x, y, z, 0)
                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                stabilizers.append(s)

                            # type 2 bulk point defect
                            elif x_incidence_type == 2 and z_incidence_type == 2:
                                # modified blue star
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                stabilizers.append(s)

                                # modified black star
                                s = self.get_star(x, y, z, 0)
                                s.append(self.lattice_to_linear(x, y - 1, z, 1, 2))
                                stabilizers.append(s)

                            elif x_incidence_type == 1 and z_incidence_type == 0:
                                # type 5 bulk point defect
                                if x_intersects == 1:
                                    # modified blue star
                                    s = self.get_star(x, y, z, 1)
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    stabilizers.append(s)

                                    # modified black star
                                    s = self.get_star(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    stabilizers.append(s)

                                # X-layer is incident but not intersecting
                                # missing defect type B
                                # type 3 line defect and normal X star
                                else:
                                    # modified black star
                                    s = self.get_star(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    stabilizers.append(s)

                                    # normal blue star
                                    stabilizers.append(self.get_star(x, y, z, 1))

                            elif x_incidence_type == 1 and z_incidence_type == 2:
                                # type 6 bulk point defect
                                if x_intersects == 1:
                                    # modified blue star
                                    s = self.get_star(x, y, z, 1)
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    stabilizers.append(s)

                                    # modified black star
                                    s = self.get_star(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y - 1, z, 1, 2))
                                    stabilizers.append(s)

                                # X-layer is incident but not intersecting
                                # missing defect type D
                                # type 5 line defect and normal X star
                                else:
                                    # modified black star
                                    s = self.get_star(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y - 1, z, 1, 2))
                                    stabilizers.append(s)

                                    # normal blue star
                                    stabilizers.append(self.get_star(x, y, z, 1))

                            elif x_incidence_type == 0 and z_incidence_type == 1:
                                # type 7 bulk point defect
                                if z_intersects == 1:
                                    # modified blue star
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                    stabilizers.append(s)

                                    # modified black star
                                    s = self.get_star(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    stabilizers.append(s)

                                    # normal red star
                                    stabilizers.append(self.get_star(x, y, z, 2))

                                # Z-layer incident but not intersecting
                                # missing defect type E
                                # type 6 line defect and normal Z star
                                else:
                                    # modified blue star
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    stabilizers.append(s)

                                    # normal black star
                                    stabilizers.append(self.get_star(x, y, z, 0))

                                    # normal red star
                                    stabilizers.append(self.get_star(x, y, z, 2))

                            elif x_incidence_type == 2 and z_incidence_type == 1:
                                # type 8 bulk point defect
                                if z_intersects == 1:
                                    # modified blue star
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                    stabilizers.append(s)

                                    # modified black star
                                    s = self.get_star(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    stabilizers.append(s)

                                    # normal red star
                                    stabilizers.append(self.get_star(x, y, z, 2))

                                # Z-layer incident but not intersecting
                                # missing defect type F
                                # type 8 line defect and normal Z star
                                else:
                                    # modified blue star
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                    stabilizers.append(s)

                                    # normal black star
                                    stabilizers.append(self.get_star(x, y, z, 0))

                                    # normal red star
                                    stabilizers.append(self.get_star(x, y, z, 2))

                            # missing defect type C
                            # THIS SHOULD BE IMPOSSIBLE
                            elif x_incidence_type == 0 and z_incidence_type == 2:
                                raise Exception("X-type stabilizer #" + str(floor(z / self.c_z) - 1) + \
                                                " and Z-type stabilizer #" + str(
                                    floor(x / self.c_x) - 1) + " has odd support.")

                            # missing defect type A
                            # THIS SHOULD BE IMPOSSIBLE
                            elif x_incidence_type == 2 and z_incidence_type == 0:
                                raise Exception("X-type stabilizer #" + str(floor(z / self.c_z) - 1) + \
                                                " and Z-type stabilizer #" + str(
                                    floor(x / self.c_x) - 1) + " has odd support.")

                            # both X and Z layers non-terminating at current Q-layer
                            else:
                                xindex = floor(z / self.c_z) - 1
                                zindex = floor(x / self.c_x) - 1

                                # check if the X and Z layers have a non-trivial y-junction at the current layer
                                # junction type 0 is non-trivial on top and trivial on bottom,
                                # junction type 2 is non-trivial on bottom and trivial on top
                                # junction type 1 is non-trivial on both sides
                                if layer in self.y_defects[xindex][zindex]:
                                    junction_type = self.y_defects[xindex][zindex][layer]

                                    # type 3 bulk point defect
                                    if junction_type == 0:
                                        # modified blue star
                                        s = self.get_star(x, y, z, 1)
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        stabilizers.append(s)

                                        # modified black star
                                        s = self.get_star(x, y, z, 0)
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)

                                        # normal red star
                                        stabilizers.append(self.get_star(x, y, z, 2))

                                    # type 4 bulk point defect
                                    elif junction_type == 2:
                                        # modified blue star
                                        s = self.get_star(x, y, z, 1)
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                        stabilizers.append(s)

                                        # modified black star
                                        s = self.get_star(x, y, z, 0)
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)

                                        # normal red star
                                        stabilizers.append(self.get_star(x, y, z, 2))

                                    # non-trivial y-junction above and below the Q layer
                                    # missing check type H
                                    else:
                                        # missing defect type H(a)
                                        if z_intersects:
                                            # modified black star
                                            s = self.get_star(x, y, z, 0)
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            stabilizers.append(s)

                                            # modified blue star
                                            s = self.get_star(x, y, z, 1)
                                            s.append(self.lattice_to_linear(x,y,z,2,2))
                                            stabilizers.append(s)

                                            # normal red star
                                            stabilizers.append(self.get_star(x, y, z, 2))

                                        # missing defect type H(b)
                                        elif x_intersects:
                                            # modified blue star
                                            s = self.get_star(x, y, z, 1)
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            stabilizers.append(s)

                                            # normal black star
                                            stabilizers.append(self.get_star(x, y, z, 0))

                                            # normal red star
                                            stabilizers.append(self.get_star(x, y, z, 2))

                                        # missing defect type H(c)
                                        else:
                                            # normal black star
                                            stabilizers.append(self.get_star(x, y, z, 0))

                                            # modified blue star
                                            s = self.get_star(x, y, z, 1)
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            stabilizers.append(s)

                                            # normal red star
                                            stabilizers.append(self.get_star(x, y, z, 2))

                                # trivial y-junction above and below the Q layer
                                # missing defect type G
                                else:
                                    # missing defect type G(a)
                                    if z_intersects == 1:
                                        # modified black star
                                        s = self.get_star(x, y, z, 0)
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)

                                        # normal blue star
                                        stabilizers.append(self.get_star(x, y, z, 1))

                                        # normal red star
                                        stabilizers.append(self.get_star(x, y, z, 2))

                                    # missing defect type G(b)
                                    elif x_intersects == 1:
                                        # normal black star
                                        stabilizers.append(self.get_star(x, y, z, 0))

                                        # modified blue star
                                        s = self.get_star(x, y, z, 1)
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        stabilizers.append(s)

                                        # normal red star
                                        stabilizers.append(self.get_star(x, y, z, 2))

                                    # missing defect type G(c)
                                    else:
                                        # normal black star
                                        stabilizers.append(self.get_star(x, y, z, 0))

                                        # normal blue star
                                        stabilizers.append(self.get_star(x, y, z, 1))

                                        # normal red star
                                        stabilizers.append(self.get_star(x, y, z, 2))

                        # normal Z line defect
                        else:
                            z_incidence_type = zjunctions[x]

                            # type 3 line defect
                            if z_incidence_type == 0:
                                # modified black star
                                s = self.get_star(x, y, z, 0)
                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                stabilizers.append(s)

                            # type 5 line defect
                            elif z_incidence_type == 2:
                                # modified black star
                                s = self.get_star(x, y, z, 0)
                                s.append(self.lattice_to_linear(x, y - 1, z, 1, 2))
                                stabilizers.append(s)

                            else:
                                # type 4 line defect
                                if z_intersects == 1:
                                    # modified black star
                                    s = self.get_star(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    stabilizers.append(s)

                                    # normal red star
                                    stabilizers.append(self.get_star(x, y, z, 2))

                                # Z-layer incident but not intersecting
                                # normal Q and Z stars
                                else:
                                    # normal black star
                                    stabilizers.append(self.get_star(x, y, z, 0))

                                    # normal red star
                                    stabilizers.append(self.get_star(x, y, z, 2))

                    # point on an X junction, but not also on a Z junction or a boundary
                    elif z in xjunctions:
                        x_intersects = self.input_xChecks[floor(z / self.c_z) - 1][layer]
                        x_incidence_type = xjunctions[z]

                        # type 6 line defect
                        if x_incidence_type == 0:
                            # modified blue star
                            s = []
                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                            s.append(self.lattice_to_linear(x, y, z, 0, 1))
                            s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                            s.append(self.lattice_to_linear(x, y, z, 1, 1))
                            stabilizers.append(s)

                            # normal black star
                            stabilizers.append(self.get_star(x, y, z, 0))

                        # type 8 line defect
                        elif x_incidence_type == 2:
                            # modified blue star
                            s = []
                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                            s.append(self.lattice_to_linear(x, y, z, 0, 1))
                            s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                            s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                            stabilizers.append(s)

                            # normal black star
                            stabilizers.append(self.get_star(x, y, z, 0))

                        else:
                            # type 7 line defect
                            if x_intersects == 1:
                                # modified blue star
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                                stabilizers.append(s)

                                # normal black star
                                stabilizers.append(self.get_star(x, y, z, 0))

                            # X-layer incident but not intersecting
                            # regular Q star and X star
                            else:
                                # normal black star
                                stabilizers.append(self.get_star(x, y, z, 0))

                                # normal blue star
                                stabilizers.append(self.get_star(x, y, z, 1))

                    # regular bulk point on Q-layer without any defects
                    # regular Q star
                    else:
                        # normal black star
                        stabilizers.append(self.get_star(x, y, z, 0))

        # this loop defines the stabilizers on the X-layers, but without the need to take into account
        # the Q-layers, which was handled by the previous loop
        for xlayer in range(self.num_xLayer):
            z = (xlayer + 1) * self.c_z
            qmin, qmax = self.span_xLayer(xlayer)

            for q in range(qmin, qmax):

                # stores a dictionary of Z-layers which is incident to the segment of the X-layer between
                # Q-layer q and q+1. The key of the dictionary is the x-coordinate of the Z-layer
                # the entry is 0 if non-intersecting (i.e. trivial y-junction) and 1 otherwise.
                z_incidences = {}
                for zlayer in range(self.num_zLayer):
                    zqmin, zqmax = self.span_zLayer(zlayer)

                    # see if a given Z layer is incident to the current segment of the X layer
                    if zqmin <= q < zqmax:
                        # see if the current incidence is an intersection
                        if q in self.y_defects[xlayer][zlayer]:
                            if self.y_defects[xlayer][zlayer][q] != 2:
                                z_incidences.update({(zlayer + 1) * self.c_x: 1})
                            else:
                                z_incidences.update({(zlayer + 1) * self.c_x: 0})
                        else:
                            z_incidences.update({(zlayer + 1) * self.c_x: 0})

                for y in range(q * self.c_y + 1, (q + 1) * self.c_y):
                    for x in range(self.x_max + 1):
                        # point on left boundary
                        if x == 0:
                            # normal (boundary) blue star
                            s = []
                            s.append(self.lattice_to_linear(x, y, z, 1, 1))
                            s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                            s.append(self.lattice_to_linear(x, y, z, 0, 1))
                            stabilizers.append(s)

                        # point on right boundary
                        elif x == self.x_max:
                            # normal (boundary) blue star
                            s = []
                            s.append(self.lattice_to_linear(x, y, z, 1, 1))
                            s.append(self.lattice_to_linear(x, y - 1, z, 1, 1))
                            s.append(self.lattice_to_linear(x - 1, y, z, 0, 1))
                            stabilizers.append(s)

                        # point on an XZ junction
                        elif x in z_incidences:
                            # type 2 line defect
                            if z_incidences[x] == 1:
                                # modified blue star
                                s = self.get_star(x, y, z, 1)
                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                stabilizers.append(s)

                                # normal red star
                                stabilizers.append(self.get_star(x, y, z, 2))

                            # the X and Z layer has trivial intersection
                            # type 1 trivial line defect
                            else:
                                # normal blue star
                                stabilizers.append(self.get_star(x, y, z, 1))

                                # normal red star
                                stabilizers.append(self.get_star(x, y, z, 2))

                        # ordinary bulk point on X-layer
                        # normal X star
                        else:
                            # normal blue star
                            stabilizers.append(self.get_star(x, y, z, 1))

        # final loop adds all the unmodified bulk star stabilizers on Z layers
        for zlayer in range(self.num_zLayer):
            x = (zlayer + 1) * self.c_x
            qmin, qmax = self.span_zLayer(zlayer)

            for q in range(qmin, qmax):
                # records the z coordinates of all incident X-layers
                x_incidences = []

                for xlayer in range(self.num_xLayer):
                    qxmin, qxmax = self.span_xLayer(xlayer)

                    # check if X layer intersects current segment of Z layer
                    if qxmin <= q < qxmax:
                        x_incidences.append((xlayer + 1) * self.c_z)

                for y in range(q * self.c_y + 1, (q + 1) * self.c_y):
                    for z in range(self.z_max + 1):
                        # normal bulk point
                        if z not in x_incidences:
                            # normal red star
                            stabilizers.append(self.get_star(x, y, z, 2))

        return stabilizers

    # c_x can now be 1
    def build_ZStabilizers(self):
        stabilizers = []

        # first we build stabilizers on Q layers
        for layer in range(self.num_qLayer):
            y = layer * self.c_y

            # these dictionaries define the positions of the junctions in the current Q layer
            # each element of the list is a pair, containing an x (or z) coordinate referencing the position
            # of the junction, as well as a index 0, 1, or 2, referencing the type of intersection.
            # index 0 denotes that the Q layer serves as a starting point, 1 denotes that the layer is intermediate
            # and 2 denotes an endpoint
            xjunctions = {}
            zjunctions = {}

            for i in range(self.num_xLayer):
                minq, maxq = self.span_xLayer(i)
                if minq <= layer <= maxq:
                    if layer == minq:
                        xjunctions.update({(i + 1) * self.c_z: 0})
                    elif layer == maxq:
                        xjunctions.update({(i + 1) * self.c_z: 2})
                    else:
                        xjunctions.update({(i + 1) * self.c_z: 1})

            for i in range(self.num_zLayer):
                minq, maxq = self.span_zLayer(i)
                if minq <= layer <= maxq:
                    if layer == minq:
                        zjunctions.update({(i + 1) * self.c_x: 0})
                    elif layer == maxq:
                        zjunctions.update({(i + 1) * self.c_x: 2})
                    else:
                        zjunctions.update({(i + 1) * self.c_x: 1})

            for z in range(-1, self.z_max + 1):
                for x in range(self.x_max):
                    # point on a Z junction
                    if x in zjunctions:
                        z_intersects = self.input_zChecks[floor(x / self.c_x) - 1][layer]

                        # check if it is a point defect
                        if z in xjunctions:
                            x_intersects = self.input_xChecks[floor(z / self.c_z) - 1][layer]

                            x_incidence_type = xjunctions[z]
                            z_incidence_type = zjunctions[x]

                            # type 1 bulk point defect
                            if x_incidence_type == 0 and z_incidence_type == 0:
                                # modified black square
                                s = self.get_square(x, y, z, 0)
                                s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                stabilizers.append(s)

                                # modified red square
                                # plaquette with two black vertical edges
                                if self.c_y == 1 and self.span_zLayer(floor(x / self.c_x) - 1)[1] == y+1:
                                    s = []
                                    s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    stabilizers.append(s)
                                # the usual case
                                else:
                                    s = []
                                    s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    stabilizers.append(s)

                                # normal blue square
                                stabilizers.append(self.get_square(x, y, z, 1))

                            # type 2 bulk point defect
                            elif x_incidence_type == 2 and z_incidence_type == 2:
                                # modified black square
                                s = self.get_square(x, y, z, 0)
                                s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                stabilizers.append(s)


                            elif x_incidence_type == 1 and z_incidence_type == 0:
                                # type 5 bulk point defect
                                if x_intersects == 1:
                                    # modified black square
                                    s = self.get_square(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    stabilizers.append(s)

                                    # modified red square
                                    # plaquette with two black vertical edges
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                        stabilizers.append(s)
                                    # the usual case
                                    else:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                        stabilizers.append(s)

                                    # normal blue square
                                    stabilizers.append(self.get_square(x, y, z, 1))

                                # type 3 line defect and incident but non-intersecting X-layer
                                else:
                                    # normal black square
                                    stabilizers.append(self.get_square(x, y, z, 0))

                                    # modified red square
                                    # plaquette with two black vertical edges
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)
                                    # the usual case
                                    else:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)

                                    # normal blue square
                                    stabilizers.append(self.get_square(x, y, z, 1))

                            elif x_incidence_type == 1 and z_incidence_type == 2:
                                # type 6 bulk point defect
                                if x_intersects == 1:
                                    # modified black square
                                    s = self.get_square(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    stabilizers.append(s)

                                    # normal blue square
                                    stabilizers.append(self.get_square(x, y, z, 1))

                                # type 5 line defect and an incident but non-intersecting X-layer
                                else:
                                    # normal black square
                                    stabilizers.append(self.get_square(x, y, z, 0))

                                    # normal blue square
                                    stabilizers.append(self.get_square(x, y, z, 1))

                            elif x_incidence_type == 0 and z_incidence_type == 1:
                                # type 7 bulk point defect
                                if z_intersects == 1:
                                    # modified black square
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 0, 0))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z + 1, 0, 0))
                                    stabilizers.append(s)

                                    # modified red square
                                    # two black vertical edges
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        stabilizers.append(s)
                                    # normal case
                                    else:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        stabilizers.append(s)

                                    # normal blue square
                                    stabilizers.append(self.get_square(x, y, z, 1))

                                # type 6 line defect and incident but non-intersecting Z-layer
                                else:
                                    # modified black square
                                    s = self.get_square(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    stabilizers.append(s)

                                    # normal blue square
                                    stabilizers.append(self.get_square(x, y, z, 1))

                                    # normal red square
                                    # boundary
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        stabilizers.append(s)
                                    # bulk
                                    else:
                                        stabilizers.append(self.get_square(x, y, z, 2))

                            elif x_incidence_type == 2 and z_incidence_type == 1:
                                # type 8 bulk point defect
                                if z_intersects == 1:
                                    # modified black square
                                    s = self.get_square(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    stabilizers.append(s)

                                    # modified red square
                                    # boundary (two black vertical edges)
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        stabilizers.append(s)
                                    # bulk
                                    else:
                                        s = self.get_square(x, y, z, 2)
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        stabilizers.append(s)

                                # type 8 line defect and incident but non-intersecting Z-layer
                                else:
                                    # modified black square
                                    s = self.get_square(x, y, z, 0)
                                    s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                    stabilizers.append(s)

                                    # normal red square
                                    # boundary
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        stabilizers.append(s)
                                    # bulk
                                    else:
                                        stabilizers.append(self.get_square(x, y, z, 2))

                            # THIS SHOULD BE IMPOSSIBLE
                            elif x_incidence_type == 0 and z_incidence_type == 2:
                                raise Exception("X-type stabilizer #" + str(floor(z / self.c_z) - 1) + \
                                                " and Z-type stabilizer #" + str(floor(x / self.c_x) - 1) + \
                                                " has odd support.")

                            # THIS SHOULD BE IMPOSSIBLE
                            elif x_incidence_type == 2 and z_incidence_type == 0:
                                raise Exception("X-type stabilizer #" + str(floor(z / self.c_z) - 1) + \
                                                " and Z-type stabilizer #" + str(floor(x / self.c_x) - 1) + \
                                                " has odd support.")

                            # both X and Z layers non-terminating
                            else:
                                xindex = floor(z / self.c_z) - 1
                                zindex = floor(x / self.c_x) - 1

                                # check if the X and Z layers have non-trivial junction at the current layer
                                # junction type 0 is non-trivial on top and trivial on bottom,
                                # junction type 2 is non-trivial on bottom and trivial on top
                                # junction type 1 is non-trivial on both sides
                                if layer in self.y_defects[xindex][zindex]:
                                    junction_type = self.y_defects[xindex][zindex][layer]

                                    # type 3 bulk point defect
                                    if junction_type == 0:
                                        # modified black square
                                        s = self.get_square(x, y, z, 0)
                                        s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                        stabilizers.append(s)

                                        # modified red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = self.get_square(x, y, z, 2)
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                            stabilizers.append(s)

                                        # normal blue square
                                        stabilizers.append(self.get_square(x, y, z, 1))

                                    # type 4 bulk point defect
                                    elif junction_type == 2:
                                        # modified black square
                                        s = self.get_square(x, y, z, 0)
                                        s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                        stabilizers.append(s)

                                        # modified red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = self.get_square(x, y, z, 2)
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            stabilizers.append(s)

                                        # normal blue square
                                        stabilizers.append(self.get_square(x, y, z, 1))

                                    # non-trivial y-junction above and below the Q layer
                                    # missing defect type H
                                    else:
                                        # missing defect type H(a)
                                        if z_intersects:
                                            # modified red square
                                            # boundary (two black vertical edges)
                                            if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                                s = []
                                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                                s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                                stabilizers.append(s)
                                            # bulk
                                            else:
                                                s = self.get_square(x, y, z, 2)
                                                s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                                stabilizers.append(s)

                                            # normal black square
                                            stabilizers.append(self.get_square(x, y, z, 0))

                                            # normal blue square
                                            stabilizers.append(self.get_square(x, y, z, 1))

                                        # missing defect type H(b)
                                        elif x_intersects:
                                            # modified black square
                                            s = self.get_square(x, y, z, 0)
                                            s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                            stabilizers.append(s)

                                            # modified red square
                                            # boundary (black vertical edges)
                                            if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                                s = []
                                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                                s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                                stabilizers.append(s)
                                            # bulk
                                            else:
                                                s = self.get_square(x, y, z, 2)
                                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                                stabilizers.append(s)

                                            # normal blue square
                                            stabilizers.append(self.get_square(x, y, z, 1))

                                        # missing defect type H(c)
                                        else:
                                            # modified red square
                                            # boundary (black vertical edges)
                                            if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                                s = []
                                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                                s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                                stabilizers.append(s)
                                            # bulk
                                            else:
                                                s = self.get_square(x, y, z, 2)
                                                s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                                stabilizers.append(s)

                                            # normal black square
                                            stabilizers.append(self.get_square(x, y, z, 0))

                                            # normal blue square
                                            stabilizers.append(self.get_square(x, y, z, 1))


                                # trivial y-junction above and below the Q layer
                                # missing defect type G
                                else:
                                    # missing defect type G(a)
                                    if z_intersects == 1:
                                        # normal black square
                                        stabilizers.append(self.get_square(x, y, z, 0))

                                        # normal blue square
                                        stabilizers.append(self.get_square(x, y, z, 1))

                                        # modified red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            s.append(self.lattice_to_linear(x,y,z,2,0))
                                            stabilizers.append(s)
                                        # bulk modified red square
                                        else:
                                            s = self.get_square(x, y, z, 2)
                                            s.append(self.lattice_to_linear(x,y,z,2,0))
                                            stabilizers.append(s)

                                    # missing defect type G(b)
                                    elif x_intersects == 1:
                                        # modified black square
                                        s = self.get_square(x, y, z, 0)
                                        s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                        stabilizers.append(s)

                                        # normal blue square
                                        stabilizers.append(self.get_square(x, y, z, 1))

                                        # modified red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = self.get_square(x, y, z, 2)
                                            stabilizers.append(s)

                                    # missing defect type G(c)
                                    else:
                                        # normal black square
                                        stabilizers.append(self.get_square(x, y, z, 0))

                                        # normal blue square
                                        stabilizers.append(self.get_square(x, y, z, 1))

                                        # modified red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = self.get_square(x, y, z, 2)
                                            stabilizers.append(s)

                        # point on Z line defect but not X
                        else:
                            z_incidence_type = zjunctions[x]

                            # type 3 line defect
                            if z_incidence_type == 0:
                                # lower boundary
                                if z == -1:
                                    # bottom boundary black square
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z + 1, 0, 0))
                                    stabilizers.append(s)

                                    # bottom boundary red square
                                    # boundary (black vertical edges)
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        stabilizers.append(s)
                                    # bulk
                                    else:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        stabilizers.append(s)

                                # upper boundary
                                elif z == self.z_max:
                                    # top boundary black square
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 0))
                                    stabilizers.append(s)

                                    # top boundary red square
                                    # boundary (black vertical edges)
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)
                                    # bulk
                                    else:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)

                                # bulk Z-defect
                                else:
                                    # normal black square
                                    stabilizers.append(self.get_square(x, y, z, 0))

                                    # red square
                                    # boundary (black vertical edges)
                                    if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)
                                    # bulk
                                    else:
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                        s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                        stabilizers.append(s)

                            # type 5 line defect
                            elif z_incidence_type == 2:
                                # lower boundary
                                if z == -1:
                                    # bottom boundary modified black square
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z + 1, 0, 0))
                                    stabilizers.append(s)

                                # upper boundary
                                elif z == self.z_max:
                                    # top boundary modified black square
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z, 0, 0))
                                    stabilizers.append(s)

                                # bulk
                                else:
                                    # normal black square
                                    stabilizers.append(self.get_square(x, y, z, 0))

                            else:
                                # type 4 line defect
                                if z_intersects == 1:
                                    # lower boundary
                                    if z == -1:
                                        # modified bottom boundary red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            stabilizers.append(s)

                                        # normal bottom boundary black square
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 0, 0))
                                        stabilizers.append(s)

                                    # upper boundary
                                    elif z == self.z_max:
                                        # modified top boundary red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            stabilizers.append(s)

                                        # normal top boundary black square
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 0, 0))
                                        stabilizers.append(s)

                                    # bulk
                                    else:
                                        # modified red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                            stabilizers.append(s)

                                        # normal black square
                                        stabilizers.append(self.get_square(x, y, z, 0))

                                # Q and Z layers incident but non-intersecting
                                # normal surface code squares
                                else:
                                    # lower boundary
                                    if z == -1:
                                        # normal bottom boundary black square
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z + 1, 0, 0))
                                        stabilizers.append(s)

                                        # normal bottom boundary red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            stabilizers.append(s)

                                    # upper boundary
                                    elif z == self.z_max:
                                        # normal top boundary black square
                                        s = []
                                        s.append(self.lattice_to_linear(x, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                                        s.append(self.lattice_to_linear(x, y, z, 0, 0))
                                        stabilizers.append(s)

                                        # normal top boundary red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            stabilizers.append(s)

                                    # bulk
                                    else:
                                        # normal black square
                                        stabilizers.append(self.get_square(x, y, z, 0))

                                        # normal red square
                                        # boundary (black vertical edges)
                                        if self.c_y == 1 and self.span_zLayer(floor(x/self.c_x)-1)[1] == y+1:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            stabilizers.append(s)
                                        # bulk
                                        else:
                                            s = []
                                            s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                            s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                            s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                            stabilizers.append(s)


                    # point on an X junction, but not also on a Z junction
                    elif z in xjunctions:
                        x_intersects = self.input_xChecks[floor(z / self.c_z) - 1][layer]
                        x_incidence_type = xjunctions[z]

                        # type 6 line defect
                        if x_incidence_type == 0:
                            # modified black square
                            s = self.get_square(x, y, z, 0)
                            s.append(self.lattice_to_linear(x, y, z, 0, 1))
                            stabilizers.append(s)

                            # normal blue square
                            stabilizers.append(self.get_square(x, y, z, 1))

                        # type 8 line defect
                        elif x_incidence_type == 2:
                            # modified black square
                            s = self.get_square(x, y, z, 0)
                            s.append(self.lattice_to_linear(x, y, z, 0, 1))
                            stabilizers.append(s)

                        else:
                            # type 7 line defect
                            if x_intersects == 1:
                                # modified black square
                                s = self.get_square(x, y, z, 0)
                                s.append(self.lattice_to_linear(x, y, z, 0, 1))
                                stabilizers.append(s)

                                # normal blue square
                                stabilizers.append(self.get_square(x, y, z, 1))

                            # X layer passing through Q but non-intersecting
                            # normal surface code stabilizers
                            else:
                                # normal black square
                                stabilizers.append(self.get_square(x, y, z, 0))

                                # normal blue square
                                stabilizers.append(self.get_square(x, y, z, 1))

                    # point not on any junction
                    # regular square on Q-layer
                    else:
                        # normal bottom boundary
                        if z == -1:
                            # bottom boundary black square
                            s = []
                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                            s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                            s.append(self.lattice_to_linear(x, y, z + 1, 0, 0))
                            stabilizers.append(s)

                        # normal top boundary
                        elif z == self.z_max:
                            # top boundary black square
                            s = []
                            s.append(self.lattice_to_linear(x, y, z, 2, 0))
                            s.append(self.lattice_to_linear(x + 1, y, z, 2, 0))
                            s.append(self.lattice_to_linear(x, y, z, 0, 0))
                            stabilizers.append(s)

                        # ordinary bulk square
                        else:
                            # normal black square
                            stabilizers.append(self.get_square(x, y, z, 0))

        # this loop defines the stabilizers on the X-layers, but without the need to take into account the Q
        # layers, which was handled by the previous loop
        for xlayer in range(self.num_xLayer):
            z = (xlayer + 1) * self.c_z
            qmin, qmax = self.span_xLayer(xlayer)

            for q in range(qmin, qmax):
                z_incidences = {}
                for zlayer in range(self.num_zLayer):
                    zqmin, zqmax = self.span_zLayer(zlayer)

                    # see if a given Z layer intersects the current segment of the X layer
                    if zqmin <= q < zqmax:
                        # see if the intersection has a trivial junction
                        if q in self.y_defects[xlayer][zlayer]:
                            if self.y_defects[xlayer][zlayer][q] != 2:
                                z_incidences.update({(zlayer + 1) * self.c_x: 1})
                            else:
                                z_incidences.update({(zlayer + 1) * self.c_x: 0})
                        else:
                            z_incidences.update({(zlayer + 1) * self.c_x: 0})

                for y in range(q * self.c_y + 1, (q + 1) * self.c_y):
                    for x in range(self.x_max):
                        if x in z_incidences:
                            # type 2 line defect
                            if z_incidences[x] == 1:

                                # modified red square
                                # boundary (black vertical edges)
                                if y + 1 == self.span_zLayer(floor(x / self.c_x) - 1)[1] * self.c_y:
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                    s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    stabilizers.append(s)
                                # bulk
                                else:
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                    s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                    s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 1))
                                    stabilizers.append(s)

                                # normal blue square
                                stabilizers.append(self.get_square(x, y, z, 1))

                            # the X and Z layer are incident but non-intersecting
                            # type 1 trivial line defect
                            else:
                                # normal blue square
                                stabilizers.append(self.get_square(x, y, z, 1))

                                # normal red square
                                # boundary (black vertical edges)
                                if y + 1 == self.span_zLayer(floor(x / self.c_x) - 1)[1] * self.c_y:
                                    s = []
                                    s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                    s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                    s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                    s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                    stabilizers.append(s)
                                # bulk
                                else:
                                    stabilizers.append(self.get_square(x, y, z, 2))

                        # ordinary bulk point on X-layer
                        # normal blue square
                        else:
                            # normal blue square
                            stabilizers.append(self.get_square(x, y, z, 1))

        # final loop adds all the bulk (unmodified) stabilizers on Z layers
        for zlayer in range(self.num_zLayer):
            x = (zlayer + 1) * self.c_x
            qmin, qmax = self.span_zLayer(zlayer)

            for q in range(qmin, qmax):
                x_incidences = []
                for xlayer in range(self.num_xLayer):
                    qxmin, qxmax = self.span_xLayer(xlayer)

                    # check if X layer intersects current segment of Z layer
                    if qxmin <= q < qxmax:
                        x_incidences.append((xlayer + 1) * self.c_z)

                for y in range(q * self.c_y + 1, (q + 1) * self.c_y):
                    for z in range(-1, self.z_max + 1):
                        # lower boundary
                        if z == -1:
                            # bottom boundary red square
                            # boundary (black vertical edges)
                            if y + 1 == self.span_zLayer(zlayer)[1] * self.c_y:
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                stabilizers.append(s)
                            # bulk
                            else:
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                stabilizers.append(s)

                        # top boundary
                        elif z == self.z_max:
                            # top boundary red square
                            # boundary (black vertical edges)
                            if y + 1 == self.span_zLayer(zlayer)[1] * self.c_y:
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                stabilizers.append(s)
                            # bulk
                            else:
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                stabilizers.append(s)

                        # normal Z-layer bulk plaquettes
                        elif z not in x_incidences:
                            # normal red square
                            # boundary (black vertical edges)
                            if y + 1 == self.span_zLayer(zlayer)[1] * self.c_y:
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 0))
                                s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                stabilizers.append(s)
                            # bulk
                            else:
                                s = []
                                s.append(self.lattice_to_linear(x, y, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y + 1, z, 2, 2))
                                s.append(self.lattice_to_linear(x, y, z + 1, 1, 2))
                                s.append(self.lattice_to_linear(x, y, z, 1, 2))
                                stabilizers.append(s)

        return stabilizers

    # returns the binary matrices defining the stabilizers
    def get_stabilizer_matrices(self):
        xchecks = self.xCheck_list
        zchecks = self.zCheck_list
        x_stabilizer = []
        z_stabilizer = []
        numq = self.num_total_qubits

        for s in xchecks:
            row = [0] * numq
            for j in s:
                row[j] = 1

            x_stabilizer.append(row)

        for s in zchecks:
            row = [0] * numq
            for j in s:
                row[j] = 1

            z_stabilizer.append(row)

        return np.asmatrix(x_stabilizer), np.asmatrix(z_stabilizer)

    # Given the index of a Z-type stabilizer, return the coordinates of the center of the plaquette that
    # defines it (will be half-integer for some coordinates)
    def get_Zstab_coord(self, stab_num):
        stab = self.zCheck_list[stab_num]
        corners = []
        for s in stab:
            endpoint1, endpoint2 = self.get_endpoints(s)
            corners.append(endpoint1)
            corners.append(endpoint2)

        corners = set(corners)
        x = y = z = 0
        for c in corners:
            x += c[0]
            y += c[1]
            z += c[2]

        return x/4, y/4, z/4



    # Get the regionID of the n-th Z stabilizer
    def get_Ztstab_regionID(self, stab_num):
        x, y, z = self.get_Zstab_coord(stab_num)

        # stabilizer lives on a Q-layer
        if y.is_integer():
            layer_num = floor(y/self.c_y)
            region_num = self.region_bins[layer_num]
            region_x = 0
            region_z = 0

            for i in self.z_defects[layer_num]:
                if x > i:
                    region_x += 1
                else:
                    break

            for j in self.x_defects[layer_num]:
                if z > j:
                    region_z += 1
                else:
                    break

            return region_num + (len(self.z_defects[layer_num])+1)*region_z + region_x
        # stabilizer lives on X-layer
        elif z.is_integer():
            x_layer_num = floor(z/self.c_z) - 1
            layer_num = x_layer_num + self.num_qLayer
            region_num = self.region_bins[layer_num]

            q = floor(y/self.c_y)
            supp = self.support_xLayer(x_layer_num)
            counter = 1
            while supp[counter] <= q:
                region_num += len(self.y_defect_on_X[x_layer_num][counter-1])+1
                counter += 1

            for i in self.y_defect_on_X[x_layer_num][counter-1]:
                if x > i:
                    region_num += 1
                else:
                    break

            return region_num
        # stabilizer lives on Z-layer
        elif x.is_integer():
            z_layer_num = floor(x / self.c_x) - 1
            layer_num = floor(x/self.c_x) - 1 + self.num_qLayer + self.num_xLayer
            region_num = self.region_bins[layer_num]

            q = floor(y/self.c_y)
            supp = self.support_zLayer(z_layer_num)
            counter = 1
            while supp[counter] <= q:
                region_num += len(self.y_defect_on_Z[z_layer_num][counter-1]) + 1
                counter += 1

            for i in self.y_defect_on_Z[z_layer_num][counter-1]:
                if z > i:
                    region_num += 1
                else:
                    break

            return region_num
        else:
            raise Exception("Something is wrong with the Z-stabilizer position.")


    # Given a region_id, returns a qubit number from a smooth boundary of the region, if it exists (it returns the central
    # qubit of the boundary) Returns None if the region does not contain a smooth boundary
    def get_boundary_qubit(self, region_id):
        # find the layer_num of the region
        layer_num = np.asarray(self.region_bins).searchsorted(region_id, 'right') - 1
        if layer_num == -1 or region_id >= self.num_regions:
            raise Exception("Invalid Region ID.")


        # local region number
        region_num = region_id - self.region_bins[layer_num]

        # region is on a Q-layer
        if layer_num < self.num_qLayer:
            modulus = len(self.z_defects[layer_num])+1
            region_x = region_num % modulus

            # left smooth boundary region
            if region_x == 0:
                region_z = floor(region_num/modulus)
                # bottom
                if region_z == 0:
                    return self.lattice_to_linear(0, layer_num*self.c_y, floor(self.x_defects[layer_num][0]/2), 2, 0)
                # top
                elif region_z == len(self.x_defects[layer_num]):
                    z0 = self.x_defects[layer_num][-1]
                    z1 = self.z_max
                    return self.lattice_to_linear(0, layer_num*self.c_y, floor((z0+z1)/2),2, 0)
                else:
                    z0 = self.x_defects[layer_num][region_z-1]
                    z1 = self.x_defects[layer_num][region_z]
                    return self.lattice_to_linear(0, layer_num*self.c_y, floor((z0+z1)/2), 2, 0)
            # right smooth boundary region
            elif region_x == modulus-1:
                region_z = floor(region_num/modulus)
                x = self.x_max
                # bottom
                if region_z == 0:
                    return self.lattice_to_linear(x, layer_num*self.c_y, floor(self.x_defects[layer_num][0]/2),2, 0)
                # top
                elif region_z == len(self.x_defects[layer_num]):
                    z0 = self.x_defects[layer_num][-1]
                    z1 = self.z_max
                    return self.lattice_to_linear(x, layer_num*self.c_y, floor((z0 + z1) / 2), 2, 0)
                else:
                    z0 = self.x_defects[layer_num][region_z - 1]
                    z1 = self.x_defects[layer_num][region_z]
                    return self.lattice_to_linear(x, layer_num*self.c_y, floor((z0 + z1) / 2), 2, 0)
            # bulk region, no smooth boundaries to match
            else:
                return None

        # region is on an X-layer
        elif layer_num < self.num_qLayer + self.num_xLayer:
            x_layer_num = layer_num-self.num_qLayer
            defects = self.y_defect_on_X[x_layer_num]
            region_x = region_num
            region_y = 0

            while region_x > len(defects[region_y])+1:
                region_x -= len(defects[region_y])+1
                region_y += 1

            # left boundary
            if region_x == 0:
                support = self.support_xLayer(x_layer_num)
                q0 = support[region_y]
                q1 = support[region_y+1]
                y = floor((q0+q1)*self.c_y/2)

                return self.lattice_to_linear(0, y, (x_layer_num+1)*self.c_z, 1, 1)
            # right boundary
            elif region_x == len(defects[region_y]):
                support = self.support_xLayer(x_layer_num)
                q0 = support[region_y]
                q1 = support[region_y + 1]
                y = floor((q0+q1)*self.c_y/2)

                return self.lattice_to_linear(self.x_max, y, (x_layer_num + 1)*self.c_z, 1, 1)

            # bulk region with no smooth boundary to match
            # returns None
            else:
                return  None

        # region is on a Z-layer, which has no smooth boundaries by definition
        # returns None
        else:
            return None

    # This function maps a Z-check to a qubit that it touches. This helps us implement
    # correction.
    def ZcheckToQubit(self, Zstab):
        x, y, z = self.get_Zstab_coord(Zstab)

        # Check is on a qubit layer. It is mapped to the qubit at its left.
        if y.is_integer():
            y = floor(y)
            x = floor(x)
            z = floor(z)

            return (x, y, z, 2, 0)
        
        # Check is on an X layer. It is mapped to the qubit at its left.
        elif z.is_integer():
            y = floor(y)
            x = floor(x)
            z = floor(z)
            return (x, y, z, 1, 1)
        
        # Check is on a Z layer. It is mapped to the qubit at its bottom.
        elif x.is_integer():
            y = floor(y)
            x = floor(x)
            z = floor(z)

            return (x, y, z, 1, 2)

        else:
            raise Exception("Something is wrong with the input stabilizer.")


    # takes in an X-logical operator of the input code, and returns
    # a corresponding X-type quasi-concatenated logical representative of the layer code
    def get_X_quasiconcatenated(self, input_logical):
        input_logical.sort()
        z = floor(self.z_max/2)
        unmatched = [[] for _ in range(self.num_zLayer)]
        qc_logical = []

        for q in input_logical:
            y = q*self.c_y

            for x in range(self.x_max+1):
                qc_logical.append(self.lattice_to_linear(x,y,z,2,0))
                if x in self.z_defects[q]:
                    if len(unmatched[floor(x/self.c_x)-1]) == 1:
                        y0 = unmatched[floor(x/self.c_x)-1].pop()
                        q0, q1 = self.span_zLayer(floor(x/self.c_x)-1)
                        supp = self.support_zLayer(floor(x/self.c_x)-1)

                        # add intermediate red edges on Z-layer
                        for i in range(y0 + 1, y):
                            qc_logical.append(self.lattice_to_linear(x, i, z, 2, 2))

                        # if Z-layer passes through ending Q-layer,
                        # we add an additional red edge at the end
                        if y < q1*self.c_y:
                            qc_logical.append(self.lattice_to_linear(x, y, z, 2, 2))

                    elif len(unmatched[floor(x/self.c_x)-1]) == 0:
                        unmatched[floor(x/self.c_x)-1].append(y)
                    else:
                        raise Exception("Something is wrong with the structure of the input logical operator.")

        if any(unmatched):
            raise Exception("Something is wrong with the structure of the input logical operator.")

        return qc_logical


    # takes in a Z-logical operator of the input code, and returns
    # a corresponding Z-type quasi-concatenated logical representative of the layer code
    def get_Z_quasiconcatenated(self, input_logical):
        input_logical.sort()
        x = floor(self.x_max/2)
        unmatched = [[] for _ in range(self.num_xLayer)]
        qc_logical = []

        for q in input_logical:
            y = q * self.c_y

            for z in range(-1, self.z_max + 1):
                qc_logical.append(self.lattice_to_linear(x, y, z, 2, 0))
                if z in self.x_defects[q]:
                    if len(unmatched[floor(z/self.c_z)-1]) == 1:
                        y0 = unmatched[floor(z/self.c_z)-1].pop()
                        for i in range(y0, y):
                            qc_logical.append(self.lattice_to_linear(x, i, z, 1, 1))
                    elif len(unmatched[floor(z/self.c_z)-1]) == 0:
                        unmatched[floor(z/self.c_z)-1].append(y)
                    else:
                        raise Exception("Something is wrong with the structure of the input logical operator.")

        if any(unmatched):
            raise Exception("Something is wrong with the structure of the input logical operator.")

        return qc_logical


    # takes in a stabilizer in a (smooth) boundary region, returns a list of qubits matching
    # that stabilizer to the (smooth) boundary
    def z_condense(self, stab_num):
        region_num = self.get_Ztstab_regionID(stab_num)
        boundary_qubit = self.get_boundary_qubit(region_num)

        if boundary_qubit is None:
            raise Exception("Stabilizer is not in a region with a boundary. Cannot be condensed.")

        boundary_stab = np.nonzero(self.zchecks[:,boundary_qubit])[0][0]

        qubits = self.z_match(stab_num, boundary_stab)
        qubits.append(boundary_qubit)
        return qubits

    # takes in two stabilizer (id) in the _same_ region
    # returns a list of qubit ids that match between the two stabilizers
    def z_match(self, stab1, stab2):
        assert self.get_Ztstab_regionID(stab1) == self.get_Ztstab_regionID(stab2), "Stabilizers must be in the same region."

        x1, y1, z1 = self.get_Zstab_coord(stab1)
        x2, y2, z2 = self.get_Zstab_coord(stab2)
        qubits = []

        # region is on a Q-layer
        if y1.is_integer():
            y = floor(y1)
            if x1 < x2:
                x = floor(x1) + 1
                z = floor(z1)
                x0 = floor(x2) + 1
                z0 = floor(z2)
            elif x2 < x1:
                x = floor(x2) + 1
                z = floor(z2)
                x0 = floor(x1) + 1
                z0 = floor(z1)
            else:
                x = floor(x1)
                x0 = floor(x2)
                z = floor(z1)
                z0 = floor(z2)

            while x < x0:
                qubits.append(self.lattice_to_linear(x,y,z,2,0))
                x += 1
            x-= 1

            if z < z0:
                z += 1
                while z <= z0:
                    qubits.append(self.lattice_to_linear(x, y, z, 0, 0))
                    z += 1
            elif z > z0:
                while z > z0:
                    qubits.append(self.lattice_to_linear(x, y, z, 0, 0))
                    z -= 1

            return qubits
        # region is on an X-layer
        elif z1.is_integer():
            z = floor(z1)
            if x1 < x2:
                x = floor(x1) + 1
                y = floor(y1)
                x0 = floor(x2) + 1
                y0 = floor(y2)
            elif x2 < x1:
                x = floor(x2) + 1
                y = floor(y2)
                x0 = floor(x1) + 1
                y0 = floor(y1)
            else:
                x = floor(x1)
                x0 = floor(x2)
                y = floor(y1)
                y0 = floor(y2)

            while x < x0:
                qubits.append(self.lattice_to_linear(x, y, z, 1, 1))
                x += 1

            x-=1
            if y < y0:
                y += 1
                while y <= y0:
                    qubits.append(self.lattice_to_linear(x, y, z, 0, 1))
                    y += 1
            elif y > y0:
                while y > y0:
                    qubits.append(self.lattice_to_linear(x, y, z, 0, 1))
                    y -= 1

            return qubits
        # region is on a Z-layer
        elif x1.is_integer():
            x = floor(x1)
            if z1 < z2:
                z = floor(z1) + 1
                y = floor(y1)
                z0 = floor(z2) + 1
                y0 = floor(y2)
            elif z2 < z1:
                z = floor(z2) + 1
                y = floor(y2)
                z0 = floor(z1) + 1
                y0 = floor(y1)
            else:
                z = floor(z1)
                z0 = floor(z2)
                y = floor(y1)
                y0 = floor(y2)

            while z < z0:
                qubits.append(self.lattice_to_linear(x, y, z, 1, 2))
                z += 1

            z-= 1
            if y < y0:
                y += 1
                while y <= y0:
                    qubits.append(self.lattice_to_linear(x, y, z, 2, 2))
                    y += 1
            elif y > y0:
                while y > y0:
                    qubits.append(self.lattice_to_linear(x, y, z, 2, 2))
                    y -= 1

            return qubits
        else:
            raise Exception("Something is wrong with the stabilizer coordinates.")

    def computeZSyndrome(self, error):
        # old
        # checks = self.zCheck_list
        # syndrome = np.zeros(len(checks))
        # for check_id, check in enumerate(checks):
        #     value = 0
        #     for qubitID in check:
        #         if error[qubitID] == 1:
        #             value += 1
        #     value = value % 2
        #     syndrome[check_id] = value
        # return syndrome

        error_np = error.view(np.ndarray)
        return np.array(list(map(lambda c: (np.sum(error_np[c]) % 2), self.zCheck_list)))


    # plots all qubits of the layer code, mostly useful for debugging
    # optionally takes in a list of Z-stabilizer ids and plots them as filled plaquettes
    def plot(self, z_stabs=None, **kwargs):
        segments = []
        c = []
        for i in range(self.num_total_qubits):
            x, y, z, edge_type, layer_type = self.linear_to_lattice(i)
            coords = [x, y, z]
            coords[edge_type] += 1
            segments.append([[x, y, z], coords])
            if layer_type == 0:
                c.append('black')
            elif layer_type == 1:
                c.append('blue')
            else:
                c.append('red')

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        lc = Line3DCollection(segments, linewidths=1, colors=c)
        ax.add_collection(lc)

        if z_stabs is not None:
            poly = []
            for s in z_stabs:
                p = []
                x, y, z = self.get_Zstab_coord(s)
                if x.is_integer():
                    p.append([x, floor(y), floor(z)])
                    p.append([x, floor(y)+1, floor(z)])
                    p.append([x, floor(y)+1, floor(z)+1])
                    p.append([x, floor(y), floor(z)+1])
                elif y.is_integer():
                    p.append([floor(x), y, floor(z)])
                    p.append([floor(x)+1, y, floor(z)])
                    p.append([floor(x+1), y, floor(z)+1])
                    p.append([floor(x), y, floor(z)+1])
                else:
                    p.append([floor(x), floor(y), z])
                    p.append([floor(x)+1, floor(y), z])
                    p.append([floor(x)+1, floor(y)+1, z])
                    p.append([floor(x), floor(y)+1, z])
                poly.append(p)

            ax.add_collection3d(Poly3DCollection(poly, **kwargs))

        ax.axes.set_xlim3d(left=-1, right=self.x_max + 1)
        ax.axes.set_ylim3d(bottom=-1, top=self.y_max + 1)
        ax.axes.set_zlim3d(bottom=-2, top=self.z_max + 2)

        ax.set_xlabel('$X$', fontsize=14)
        ax.set_ylabel('$Y$', fontsize=14)
        ax.set_zlabel('$Z$', fontsize=14)

        plt.show()


    # plot the stabilizers, mainly for debugging purposes
    def plot_stabilizers(self, stabilizers):
        segments = []
        c = []
        for s in stabilizers:
            for q in s:
                x, y, z, edge_type, layer_type = self.linear_to_lattice(q)
                coords = [x, y, z]
                coords[edge_type] += 1
                segments.append([[x, y, z], coords])
                if layer_type == 0:
                    c.append('black')
                elif layer_type == 1:
                    c.append('blue')
                else:
                    c.append('red')

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        lc = Line3DCollection(segments, linewidths=1, colors=c)
        ax.add_collection(lc)

        ax.axes.set_xlim3d(left=-1, right=self.x_max + 1)
        ax.axes.set_ylim3d(bottom=-2, top=self.y_max + 1)
        ax.axes.set_zlim3d(bottom=-2, top=self.z_max + 2)

        ax.set_xlabel('$X$', fontsize=14)
        ax.set_ylabel('$Y$', fontsize=14)
        ax.set_zlabel('$Z$', fontsize=14)

        plt.show()


    # plots the quasi-concatenated logical operators, main for debugging purposes
    def plot_logical(self, logical):
        segments = []

        for q in logical:
            x, y, z, edge_type, layer_type = self.linear_to_lattice(q)
            coords = [x, y, z]
            coords[edge_type] += 1
            segments.append([[x, y, z], coords])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        lc = Line3DCollection(segments, linewidths=1, color='black')
        ax.add_collection(lc)

        ax.axes.set_xlim3d(left=-1, right=self.x_max + 1)
        ax.axes.set_ylim3d(bottom=-1, top=self.y_max + 1)
        ax.axes.set_zlim3d(bottom=-2, top=self.z_max + 2)

        ax.set_xlabel('$X$', fontsize=14)
        ax.set_ylabel('$Y$', fontsize=14)
        ax.set_zlabel('$Z$', fontsize=14)

        plt.show()