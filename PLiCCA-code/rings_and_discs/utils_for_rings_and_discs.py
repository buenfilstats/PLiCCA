import numpy as np
#import torch.nn.functional as F
#import torch.nn.utils.parametrize as parametrize
import matplotlib.pyplot as plt

from scipy.stats import ortho_group

def generate_ring_image(r_1, r_2, num_pixels=100, aspect_ratio=1.0):
    """
    Generate a grey-scale image of a ring parameterized by r_1 and r_2.

    Parameters:
        r_1 (float): Radius of the hole inside the disc.
        r_2 (float): Thickness of the disc.
        num_pixels (int): Size of the image (num_pixels x num_pixels).
        aspect_ratio (float): Ellipse aspect ratio along x:
            1.0 = circle, >1 = stretched horizontally, <1 = stretched vertically.

    Returns:
        np.ndarray: A num_pixels x num_pixels matrix representing the ring image.
    """
    # Create a grid of coordinates
    x = np.linspace(-1, 1, num_pixels)
    y = np.linspace(-1, 1, num_pixels)
    xx, yy = np.meshgrid(x, y)
    #radius = np.sqrt(xx**2 + yy**2)
    radius = np.sqrt((xx / aspect_ratio) ** 2 + yy ** 2)
    # Create the ring image
    ring_image = np.zeros((num_pixels, num_pixels))
    ring_image[(radius >= r_1) & (radius <= r_1 + r_2)] = 1.0

    return ring_image

def display_image(image_matrix):
    """
    Display a grey-scale image from its representative matrix.

    Parameters:
        image_matrix (np.ndarray): A matrix representing the image.
    """
    image_matrix = np.clip(image_matrix, 0.0, 1.0)
    print(image_matrix)
    plt.imshow(image_matrix, cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.show()

def generate_image_dataset_and_X(N, p, X_decoder_var, Y_decoder_var, num_pixels=40, 
                                         disc_radius_lower_bd=0.2, disc_radius_upper_bd=0.4, 
                                         ring_r_1_lower_bd=0.1, ring_r_1_upper_bd=0.3, 
                                         ring_r_2_lower_bd=0.1, ring_r_2_upper_bd=0.3):
    """
    Generate a dataset of images containing discs and rings, along with X.

    Parameters:
        N (int): Total number of images to generate (should be even).
        p (int): Length of the vector X.
        X_decoder_var (float): Variance of the Gaussian noise for the last p-2 entries of X.
        Y_decoder_var (float): Variance of the Gaussian noise for the generated images.
        num_pixels (int): Size of the images (num_pixels x num_pixels).
        disc_radius_lower_bd (float): Lower bound for disc radius.
        disc_radius_upper_bd (float): Upper bound for disc radius.
        ring_r_1_lower_bd (float): Lower bound for ring inner radius.
        ring_r_1_upper_bd (float): Upper bound for ring inner radius.
        ring_r_2_lower_bd (float): Lower bound for ring thickness.
        ring_r_2_upper_bd (float): Upper bound for ring thickness.

    Returns:
        list: A list of N images (numpy arrays).
        list: A list of labels (0 for disc, 1 for ring).
        list: A list of N vectors X, each of length p.
        list: A list of N vectors Z, each containing the parameters used to generate the images.
    """
    assert N % 2 == 0, "N must be even to generate equal numbers of discs and rings."
    
    labels = []
    Z_vectors = []
    X_vectors = []
    images = []
    Z_original = []
    # Generate N/2 discs
    for _ in range((N // 4)):
        r_1 = 0
        r_2 = np.random.uniform(disc_radius_lower_bd, disc_radius_upper_bd)

        Z = np.array([r_1, r_2])

        Z_rotated = np.zeros(2)
        Z_rotated[0] = Z[0]*0.5 + Z[1]*0.5
        Z_rotated[1] = Z[0]*0.5 - Z[1]*0.5
        # Generate X vector
        X = np.concatenate([Z_rotated, np.zeros(p - 2)]) + np.random.normal(0, np.sqrt(X_decoder_var), p)  # First two entries are Z, rest are zeros

        # Generate Y image (matrix representation of the image)
        disc_image = generate_ring_image(Z[0], Z[1], num_pixels) + np.random.normal(0, np.sqrt(Y_decoder_var), (num_pixels, num_pixels))

        labels.append(0)  # Label 0 for disc
        Z_vectors.append(Z_rotated)
        Z_original.append(Z)
        X_vectors.append(X)
        images.append(disc_image)
    
    # Generate N/2 rings
    for _ in range((N // 4) * 3):
        r_1 = np.random.uniform(ring_r_1_lower_bd, ring_r_1_upper_bd)
        r_2 = np.random.uniform(ring_r_2_lower_bd, ring_r_2_upper_bd)

        Z = np.array([r_1, r_2])
        Z_rotated = np.zeros(2)
        Z_rotated[0] = Z[0]*0.5 + Z[1]*0.5
        Z_rotated[1] = Z[0]*0.5 - Z[1]*0.5
        # Generate X vector
        X = np.concatenate([Z_rotated, np.zeros(p - 2)]) + np.random.normal(0, np.sqrt(X_decoder_var), p)  # First two entries are Z, rest are zeros

        # Generate Y image (matrix representation of the image)
        ring_image = generate_ring_image(Z[0], Z[1], num_pixels) + np.random.normal(0, np.sqrt(Y_decoder_var), (num_pixels, num_pixels))

        labels.append(1)  # Label 1 for ring
        Z_vectors.append(Z_rotated)
        X_vectors.append(X)
        images.append(ring_image)
        Z_original.append(Z)
    
    return images, labels, X_vectors, Z_vectors, Z_original
def generate_image_dataset_and_X_1_d(N, p, X_decoder_var, Y_decoder_var, num_pixels=40, 
                                         disc_radius_lower_bd=0.2, disc_radius_upper_bd=0.4, 
                                         ring_r_1_lower_bd=0.1, ring_r_1_upper_bd=0.3, 
                                         ring_r_2_lower_bd=0.1, ring_r_2_upper_bd=0.3):
    """
    Generate a dataset of images containing discs and rings, along with X.

    Parameters:
        N (int): Total number of images to generate (should be even).
        p (int): Length of the vector X.
        X_decoder_var (float): Variance of the Gaussian noise for the last p-2 entries of X.
        Y_decoder_var (float): Variance of the Gaussian noise for the generated images.
        num_pixels (int): Size of the images (num_pixels x num_pixels).
        disc_radius_lower_bd (float): Lower bound for disc radius.
        disc_radius_upper_bd (float): Upper bound for disc radius.
        ring_r_1_lower_bd (float): Lower bound for ring inner radius.
        ring_r_1_upper_bd (float): Upper bound for ring inner radius.
        ring_r_2_lower_bd (float): Lower bound for ring thickness.
        ring_r_2_upper_bd (float): Upper bound for ring thickness.

    Returns:
        list: A list of N images (numpy arrays).
        list: A list of labels (0 for disc, 1 for ring).
        list: A list of N vectors X, each of length p.
        list: A list of N vectors Z, each containing the parameters used to generate the images.
    """
    assert N % 2 == 0, "N must be even to generate equal numbers of discs and rings."
    
    labels = []
    Z_vectors = []
    X_vectors = []
    images = []
    Z_original = []

    # Generate N/2 discs
    for _ in range((N // 2)):
        r_1 = 0
        r_2 = np.random.uniform(disc_radius_lower_bd, disc_radius_upper_bd)

        Z = np.array([r_1, r_2])

        Z_rotated = np.zeros(2)
        Z_rotated[0] = Z[0]*0.5 + Z[1]*0.5
        Z_rotated[1] = Z[0]*0.5 - Z[1]*0.5
        # Generate X vector
        X = np.concatenate([Z_rotated, np.zeros(p - 2)]) + np.random.normal(0, np.sqrt(X_decoder_var), p)  # First two entries are Z, rest are zeros

        # Generate Y image (matrix representation of the image)
        disc_image = generate_ring_image(Z[0], Z[1], num_pixels) + np.random.normal(0, np.sqrt(Y_decoder_var), (num_pixels, num_pixels))

        labels.append(0)  # Label 0 for disc
        Z_vectors.append(Z_rotated)
        X_vectors.append(X)
        images.append(disc_image)
        Z_original.append(Z)

    # Generate N/2 rings
    for _ in range(N // 2):
        r_1 = np.random.uniform(ring_r_1_lower_bd, ring_r_1_upper_bd)
        r_2 = np.random.uniform(ring_r_2_lower_bd, ring_r_2_upper_bd)

        Z = np.array([r_1, r_2])
        Z_rotated = np.zeros(2)
        Z_rotated[0] = Z[0]*0.5 + Z[1]*0.5
        Z_rotated[1] = Z[0]*0.5 - Z[1]*0.5
        # Generate X vector
        X = np.concatenate([Z_rotated, np.zeros(p - 2)]) + np.random.normal(0, np.sqrt(X_decoder_var), p)  # First two entries are Z, rest are zeros

        # Generate Y image (matrix representation of the image)
        ring_image = generate_ring_image(Z[0], Z[1], num_pixels) + np.random.normal(0, np.sqrt(Y_decoder_var), (num_pixels, num_pixels))

        labels.append(1)  # Label 1 for ring
        Z_vectors.append(Z_rotated)
        X_vectors.append(X)
        images.append(ring_image)
        Z_original.append(Z)

    
    return images, labels, X_vectors, Z_vectors, Z_original

# r_1 is the radius of the hole in the ring
# r_2 is the thickness of the ring
# r_3 is a scaling factor for the constrast of the image
# r_4 is the aspect ratio of the ellipse
def generate_image_dataset_from_fixed_params(Y_decoder_var, r_1_array, r_2_array, r_3_array, r_4_array, num_pixels=40):

    N = r_1_array.shape[0]
    images = []
    
    for i in range(N):
        r_1 = r_1_array[i]
        r_2 = r_2_array[i]
        r_3 = r_3_array[i]
        r_4 = r_4_array[i]
        Z = np.array([r_1, r_2])
        
        # Generate Y image (matrix representation of the image)
        ring_image = generate_ring_image(Z[0], Z[1], num_pixels, aspect_ratio=r_4)*r_3 + np.random.normal(0, np.sqrt(Y_decoder_var), (num_pixels, num_pixels))

        images.append(ring_image)
    
    return images

def generate_images_and_X_known_cvs_dataset(N, p, q, lams, thetas, etas,num_pixels,image_noise_var,
                           ring_r_1_lower_bd, ring_r_1_upper_bd,
                           disc_r_2_lower_bd, disc_r_2_upper_bd,
                           constrast_param_lower_bd, constrast_param_upper_bd,
                           aspect_ratio_lower_bd, aspect_ratio_upper_bd):
    output_dict = generate_classical_CCA_data(p=p,q= q, N=N, covtype = "id", canvecstruc = "na", k1=1, k2=2, lams=lams, SigmaX=None, SigmaY=None, thetas=thetas, etas=etas)
    from sklearn.preprocessing import StandardScaler

    #Sigma = output_dict["Sigma"]
    #print(Sigma)
    X = output_dict["X"]
    Z_standard = output_dict["Y"]
    SigmaXZ = output_dict["SigmaXY"]
    #print(SigmaXZ)
    W = np.zeros((N, 2))
    W[:,0:2] = Z_standard[:,0:2]
    # W[:, 0] = Z_standard[:, 0]**2 + Z_standard[:, 1]**2 - 2
    # W[:, 1] = np.arctan2(Z_standard[:, 1], Z_standard[:, 0])

    scaler_W = StandardScaler()
    W = scaler_W.fit_transform(W)
    Z = np.zeros((N, q))

    parameter_0_diam = ring_r_1_upper_bd - ring_r_1_lower_bd
    scaling_factor_0 = parameter_0_diam/(2*3)
    Z[:,0] = W[:,0]*scaling_factor_0 + parameter_0_diam/2 + ring_r_1_lower_bd
    # Round any Z[:,0] below ring_r_1_lower_bd to be ring_r_1_lower_bd
    Z[:,0] = np.maximum(Z[:,0], ring_r_1_lower_bd)

    parameter_1_diam = disc_r_2_upper_bd - disc_r_2_lower_bd
    scaling_factor_1 = parameter_1_diam/(2*3)
    Z[:,1] = W[:,1]*scaling_factor_1 + parameter_1_diam/2 + disc_r_2_lower_bd
    Z[:,1] = np.maximum(Z[:,1], disc_r_2_lower_bd)

    parameter_2_diam = constrast_param_upper_bd - constrast_param_lower_bd
    scaling_factor_2 = parameter_2_diam/(2*3)
    Z[:,2] = Z_standard[:,2]*scaling_factor_2 + parameter_2_diam/2 + constrast_param_lower_bd
    Z[:,2] = np.maximum(Z[:,2], constrast_param_lower_bd)
    parameter_3_diam = aspect_ratio_upper_bd - aspect_ratio_lower_bd
    scaling_factor_3 = parameter_3_diam/(2*3)
    Z[:,3] = Z_standard[:,3]*scaling_factor_3 + parameter_3_diam/2 + aspect_ratio_lower_bd
    Z[:,3] = np.maximum(Z[:,3], aspect_ratio_lower_bd)

    # W = np.zeros((N, 2))
    # #W[:,0:2] = Z_standard[:,0:2]
    # W[:, 0] = Z_standard[:, 2]**2 + Z_standard[:, 3]**2 - 2
    # W[:, 1] = np.arctan2(Z_standard[:, 3], Z_standard[:, 2])

    # scaler_W = StandardScaler()
    # W = scaler_W.fit_transform(W)
    # Z = np.zeros((N, q))

    # parameter_0_diam = ring_r_1_upper_bd - ring_r_1_lower_bd
    # scaling_factor_0 = parameter_0_diam/(2*3)
    # Z[:,0] = Z_standard[:,0]*scaling_factor_0 + parameter_0_diam/2 + ring_r_1_lower_bd
    # # Round any Z[:,0] below ring_r_1_lower_bd to be ring_r_1_lower_bd
    # Z[:,0] = np.maximum(Z[:,0], ring_r_1_lower_bd)

    # parameter_1_diam = disc_r_2_upper_bd - disc_r_2_lower_bd
    # scaling_factor_1 = parameter_1_diam/(2*3)
    # Z[:,1] = Z_standard[:,1]*scaling_factor_1 + parameter_1_diam/2 + disc_r_2_lower_bd
    # Z[:,1] = np.maximum(Z[:,1], disc_r_2_lower_bd)

    # parameter_2_diam = constrast_param_upper_bd - constrast_param_lower_bd
    # scaling_factor_2 = parameter_2_diam/(2*3)
    # Z[:,2] = W[:,0]*scaling_factor_2 + parameter_2_diam/2 + constrast_param_lower_bd
    # Z[:,2] = np.maximum(Z[:,2], constrast_param_lower_bd)
    # parameter_3_diam = aspect_ratio_upper_bd - aspect_ratio_lower_bd
    # scaling_factor_3 = parameter_3_diam/(2*3)
    # Z[:,3] = W[:,1]*scaling_factor_3 + parameter_3_diam/2 + aspect_ratio_lower_bd
    # Z[:,3] = np.maximum(Z[:,3], aspect_ratio_lower_bd)
    # Assuming SigmaXY is already computed
    # Perform Singular Value Decomposition
    U, singular_values, Vt = np.linalg.svd(SigmaXZ)

    # Print the singular values
    print("Singular values of SigmaXY, which should be equal to lams if thetas and etas satisfy the SigmaX and SigmaY normaliziation condition:")
    print(singular_values[0:len(lams)])
    print(lams)
    images = generate_image_dataset_from_fixed_params(image_noise_var, Z[:,0], Z[:,1], Z[:,2], Z[:,3], num_pixels=num_pixels)
    Y = np.array([image.flatten() for image in images])


    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X) #standardize X

    return {"X":X, "images":images, "Y":Y, "Z":Z, "Z_standard":Z_standard, "W":W}

def plot_image_dataset(images, labels):
    # Separate the images into discs and rings
    # compute number of discs
    num_discs = sum(1 for label in labels if label == 0)
    disc_images = [image for image, label in zip(images, labels) if label == 0]
    ring_images = [image for image, label in zip(images, labels) if label == 1]

    # only plot discs if there are any
    if num_discs > 0:
        # Calculate grid dimensions for disc images
        disc_grid_size = int(np.ceil(len(disc_images) ** 0.5))
        fig, axs = plt.subplots(disc_grid_size, disc_grid_size, figsize=(5, 5))
        fig.suptitle("Disc Images")
        for i, disc_image in enumerate(disc_images):
            ax = axs[i // disc_grid_size, i % disc_grid_size]
            ax.imshow(disc_image, cmap='gray', origin='lower',vmin=0.0, vmax=1.0)
            ax.axis('off')

        # Hide unused subplots for discs
        for i in range(len(disc_images), disc_grid_size * disc_grid_size):
            axs[i // disc_grid_size, i % disc_grid_size].axis('off')

    # Calculate grid dimensions for ring images
    ring_grid_size = int(np.ceil(len(ring_images) ** 0.5))
    fig, axs = plt.subplots(ring_grid_size, ring_grid_size, figsize=(5, 5))
    #fig.suptitle("Ring Images")
    for i, ring_image in enumerate(ring_images):
        ax = axs[i // ring_grid_size, i % ring_grid_size]
        ax.imshow(ring_image, cmap='gray', origin='lower',vmin=0.0, vmax=1.0)
        ax.axis('off')

    # Hide unused subplots for rings
    for i in range(len(ring_images), ring_grid_size * ring_grid_size):
        axs[i // ring_grid_size, i % ring_grid_size].axis('off')

    plt.show()

# I want to be able to compare two images side by side, given what sample numbers they are
def compare_images(images, sample_num_1, sample_num_2):
    """
    Compare two images side by side.
    
    Parameters:
        images (list): List of images (numpy arrays).
        sample_num_1 (int): Index of the first image to compare.
        sample_num_2 (int): Index of the second image to compare.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(images[sample_num_1], cmap='gray', origin='lower')
    axs[0].set_title(f'Sample {sample_num_1}')
    axs[0].axis('off')
    
    axs[1].imshow(images[sample_num_2], cmap='gray', origin='lower')
    axs[1].set_title(f'Sample {sample_num_2}')
    axs[1].axis('off')
    
    plt.show()


# Let's see how well the VAE reconstructs the images
def plot_reconstruction(Y, Yhat,num_samples=None):
    """
    Plot original and reconstructed images.
    
    Parameters:
        Y (np.ndarray): Original Y data (images).
        Yhat (np.ndarray): Reconstructed Y data (images).
        num_samples (int): Number of samples to plot.
    """
    if num_samples is None or num_samples > len(Y):
        num_samples = len(Y)

    fig, axs = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    num_pixels = int(np.floor(np.sqrt(Y.shape[1])))
    indices = np.random.choice(len(Y), size=num_samples, replace=False)
    i = 0
    for j in indices:

        # Plot original image
        axs[i, 0].imshow(Y[j].reshape(num_pixels, num_pixels), cmap='gray', origin='lower',vmin=0.0, vmax=1.0)
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')
        
        # Plot reconstructed image
        axs[i, 1].imshow(Yhat[j].reshape(num_pixels, num_pixels), cmap='gray', origin='lower',vmin=0.0, vmax=1.0)
        axs[i, 1].set_title('Reconstructed Image')
        axs[i, 1].axis('off')
        i += 1
    plt.tight_layout()
    plt.show()


#generates X and Y which are jointly gaussian with known canonical vectors, covariance matrices and canonical correlations.
def generate_classical_CCA_data(p, q, N, covtype, canvecstruc, k1, k2, lams, SigmaX=None, SigmaY=None, thetas=None, etas=None):
    
    import numpy as np
    from scipy.stats import ortho_group
    from scipy.linalg import sqrtm
    from scipy.linalg import block_diag
    r = len(lams)
    if covtype == "custom":
        Sigma1 = SigmaX
        Sigma2 = SigmaY
    elif covtype == "id":
        Sigma1 = np.eye(p)
        Sigma2 = np.eye(q)
    else:
        print("Error: covtype not specified correctly")
        return None

    if canvecstruc == "both_random_sparse":
        uset = np.random.choice(np.arange(p), size=k1 * r, replace=False)
        vset = np.random.choice(np.arange(q), size=k2 * r, replace=False)
        us = np.zeros((p, r))
        vs = np.zeros((q, r))

        for i in range(r):
            useti = uset[i * k1: (i + 1) * k1]
            us[useti, i] = 1
            u = us[:, i]
            norm = np.sqrt(u @ Sigma1 @ u)
            us[:, i] = u / norm

            vseti = vset[i * k2: (i + 1) * k2]
            vs[vseti, i] = 1
            v = vs[:, i]
            norm = np.sqrt(v @ Sigma2 @ v)
            vs[:, i] = v / norm

    elif canvecstruc == "theta_random_sparse":
        uset = np.random.choice(np.arange(p), size=k1 * r, replace=False)
        us = np.zeros((p, r))
        vs = ortho_group.rvs(q)[:, :r]
        vs = sqrtm(np.linalg.inv(Sigma2)) @ vs

        for i in range(r):
            useti = uset[i * k1: (i + 1) * k1]
            us[useti, i] = 1
            u = us[:, i]
            norm = np.sqrt(u @ Sigma1 @ u)
            us[:, i] = u / norm

    elif canvecstruc == "theta_group_sparse":
        uset = np.random.choice(np.arange(p), size=k1, replace=False)
        uset = np.sort(uset)
        vecs = ortho_group.rvs(k1)[:, :r]
        Sigma1sub = Sigma1[np.ix_(uset, uset)]
        uspre = sqrtm(np.linalg.inv(Sigma1sub)) @ vecs
        us = np.zeros((p, r))
        vs = ortho_group.rvs(q)[:, :r]
        vs = sqrtm(np.linalg.inv(Sigma2)) @ vs

        for i in range(r):
            us[uset, i] = uspre[:, i]

    elif canvecstruc == "both_group_sparse":
        uset = np.random.choice(np.arange(p), size=k1, replace=False)
        uset = np.sort(uset)
        vecs1 = ortho_group.rvs(k1)[:, :r]
        Sigma1sub = Sigma1[np.ix_(uset, uset)]
        uspre = sqrtm(np.linalg.inv(Sigma1sub)) @ vecs1
        us = np.zeros((p, r))
        vs = ortho_group.rvs(q)[:, :r]
        vs = sqrtm(np.linalg.inv(Sigma2)) @ vs

        for i in range(r):
            us[uset, i] = uspre[:, i]

        vset = np.random.choice(np.arange(q), size=k2, replace=False)
        vset = np.sort(vset)
        vecs2 = ortho_group.rvs(k2)[:, :r]
        Sigma2sub = Sigma2[np.ix_(vset, vset)]
        vspre = sqrtm(np.linalg.inv(Sigma2sub)) @ vecs2
        vs = np.zeros((q, r))
        for i in range(r):
            vs[vset, i] = vspre[:, i]



    elif canvecstruc == "theta_group_sparse_smaller_SigmaX":
        vs = ortho_group.rvs(q)[:, :r]
        vs = sqrtm(np.linalg.inv(Sigma2)) @ vs
        p1 = Sigma1.shape[0]

        if p1 == p:
            us = ortho_group.rvs(p)[:, :r]

            for i in range(r):
                u = us[:, i]
                norm = np.sqrt(u @ Sigma1 @ u)
                us[:, i] = u / norm

        elif p1 < p:
            ustemp = ortho_group.rvs(p1)[:, :r]
            ustemp = sqrtm(np.linalg.inv(Sigma1)) @ ustemp
            us = np.zeros((p, r))
            us[:p1, :] = ustemp
            Sigma1temp = Sigma1.copy()
            Sigma1 = np.eye(p)
            Sigma1[:p1, :p1] = Sigma1temp
    elif thetas is not None and etas is not None:
        us = thetas
        vs = etas
    else:
        print("Error: canvecstruc not correctly specified")
        return None

    partial = np.zeros((p, q))
    for i in range(r):
        partial += lams[i] * np.outer(us[:, i], vs[:, i])

    Sigma12 = Sigma1 @ partial @ Sigma2

    Sigma = block_diag(Sigma1, Sigma2)
    Sigma[:p, p:] = Sigma12
    Sigma[p:, :p] = Sigma12.T

    XandY = np.random.multivariate_normal(np.zeros(p + q), Sigma, size=N)
    X = XandY[:, :p]
    Y = XandY[:, p:]

    return {"X": X, "Y": Y, "thetas": us, "etas": vs, "SigmaXY": Sigma12, "SigmaX": Sigma1, "SigmaY": Sigma2,"Sigma": Sigma}