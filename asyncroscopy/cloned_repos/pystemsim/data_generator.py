# Description: This file contains the functions to generate synthetic data for the neural network training.
# By Austin Houston
# Date: 02/28/2024
# Updated: 05/10/2024

import dask
import numpy as np
import random
import sidpy
import dask.array as da
import scipy.special as sp
from scipy.ndimage import zoom, gaussian_filter
from skimage.draw import disk
from ase import Atoms
from ase.neighborlist import NeighborList
from scipy.fft import fft2, ifft2
import pyTEMlib.probe_tools as pt


def make_holes(atoms: Atoms, n_holes: int, hole_size: float) -> Atoms:
    """
    Create holes in an Atoms object by deleting atoms around randomly selected positions.

    Parameters:
    - atoms (ase.Atoms): The input Atoms object.
    - n_holes (int): The number of holes to create.
    - hole_size (float): The radius of each hole.

    Returns:
    - ase.Atoms: The modified Atoms object with holes.
    """
    # Step 1: Randomly select n_holes atoms
    num_atoms = len(atoms)
    selected_indices = random.sample(range(num_atoms), n_holes)

    # Step 2: Find and delete atoms within radius hole_size
    for index in selected_indices:
        # Get the position of the selected atom
        pos = atoms[index].position

        # Create a NeighborList to find atoms within hole_size
        cutoffs = [hole_size / 2] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        # Find atoms within hole_size around the selected atom
        indices, offsets = nl.get_neighbors(index)
        indices = indices.tolist()

        # Add the selected atom itself to the list of atoms to be deleted
        indices.append(index)

        # Delete atoms by their indices
        atoms = atoms[[atom.index for atom in atoms if atom.index not in indices]]

    return atoms

def rotate_xtal(xtal, angle):
    # pad for worst case and rotate
    padded = xtal * (2, 2, 1)
    padded.rotate('z', angle, 'com')

    # crop to original cell
    cell = xtal.cell
    positions = padded.get_positions()[:, :2]
    inv_cell = np.linalg.inv(cell[:2, :2])
    frac = positions @ inv_cell - 0.5
    mask = np.all((frac >= 0) & (frac < 1), axis=1)

    # creat the new xtal object
    xtal_cropped = padded[mask].copy()
    xtal_cropped.set_cell(cell, scale_atoms=False)
    xtal_cropped.set_scaled_positions(np.hstack([frac[mask], padded.get_scaled_positions()[mask, 2:3]]))

    return xtal_cropped

def sub_pix_gaussian(size=10, sigma=0.2, dx=0.0, dy=0.0):
    # returns sub-pix shifted gaussian
    coords = np.arange(size) - (size - 1) / 2.0
    x, y = np.meshgrid(coords, coords)
    g = np.exp(-(((x + dx) ** 2 + (y + dy) ** 2) / (2 * sigma**2)))
    g /= g.max()
    return g

def create_pseudo_potential(xtal, pixel_size, sigma, bounds, atom_frame=11):
    # Create empty image
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    pixels_x = int((x_max - x_min) / pixel_size)
    pixels_y = int((y_max - y_min) / pixel_size)
    potential_map = np.zeros((pixels_x, pixels_y))
    padding = atom_frame  # to avoid edge effects
    potential_map = np.pad(potential_map, padding, mode='constant', constant_values=0.0)

    # Map of atomic numbers - i.e. scattering intensity
    atomic_numbers = xtal.get_atomic_numbers()
    positions = xtal.get_positions()[:, :2]

    mask = ((positions[:, 0] >= x_min) & (positions[:, 0] < x_max) & (positions[:, 1] >= y_min) & (positions[:, 1] < y_max))
    positions = positions[mask]
    atomic_numbers = atomic_numbers[mask]

    for pos, atomic_number in zip(positions, atomic_numbers):
        x,y = np.round(pos/pixel_size)
        dx,dy = pos - np.round(pos)
  
        single_atom = sub_pix_gaussian(size=atom_frame, sigma=sigma, dx=dx, dy=dy) * atomic_number
        potential_map[int(x+padding+dx-padding//2-1):int(x+padding+dx+padding//2),int(y+padding+dy-padding//2-1):int(y+padding+dy+padding//2)] += single_atom
    potential_map = potential_map[padding:-padding, padding:-padding]
    normalized_map = potential_map / np.max(potential_map)

    # make a sidpy dataset
    dset = sidpy.Dataset.from_array(normalized_map, name = 'Scattering Potential')
    dset.data_type = 'image'
    dset.units = 'A.U.'
    dset.quantity = 'Scattering cross-section'
    dset.set_dimension(0, sidpy.Dimension(pixel_size * np.arange(pixels_x),
                        name='x', units='Å', quantity='Length',dimension_type='spatial'))
    dset.set_dimension(1, sidpy.Dimension(pixel_size * np.arange(pixels_y),
                        name='y', units='Å', quantity='Length',dimension_type='spatial'))

    return dset


def get_masks(xtal, pixel_size=0.1, radius=3, axis_extent=None, mode='one_hot'):
    positions = xtal.get_positions()[:, :2]
    atomic_numbers = xtal.get_atomic_numbers()
    _, inverse_indices = np.unique(atomic_numbers, return_inverse=True)
    atom_ids = inverse_indices + 1  # the background pixels will be labeled as 0
    unique_atom_ids = np.unique(atom_ids)

    # Determine image size
    if axis_extent is not None:
        xmin, xmax, ymin, ymax = axis_extent
    else:
        xmin, xmax = np.min(positions[:, 0]), np.max(positions[:, 0])
        ymin, ymax = np.min(positions[:, 1]), np.max(positions[:, 1])
    img_height = int((ymax - ymin) / pixel_size)
    img_width = int((xmax - xmin) / pixel_size)

    master_mask = np.zeros((len(unique_atom_ids), img_height, img_width), dtype=np.uint8)
    
    def create_mask_for_atom(atom_id):
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        atom_mask = (atom_ids == atom_id)
        atom_positions = positions[atom_mask]

        # Make mask 1 in radius around each atom
        for x, y in atom_positions:
            x_pixel = int((x - xmin) / pixel_size)
            y_pixel = int((y - ymin) / pixel_size)
            rr, cc = disk((y_pixel, x_pixel), radius, shape=mask.shape)
            mask[rr, cc] = 1
        master_mask[atom_id - 1, mask == 1] = 1

    # Parallelize the mask creation
    tasks = [dask.delayed(create_mask_for_atom)(atom_id) for atom_id in unique_atom_ids]
    dask.compute(*tasks)

    if mode.lower() == 'one_hot':
        num_masks = unique_atom_ids.size + 1  # include background
        background_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        background_mask[(np.sum(master_mask, axis=0) == 0)] = 1
        masks = np.stack([background_mask] + [master_mask[i] for i in range(len(unique_atom_ids))], axis=0)
        return masks

    elif mode.lower() == 'binary':
        sum_masks = np.sum(master_mask, axis=0)
        final_mask = np.where(sum_masks > 0, 1, 0)
        return final_mask

    elif mode.lower() == 'integer':
        final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for i, mask in enumerate(master_mask):
            final_mask[mask == 1] = i + 1
        return final_mask

    else:
        raise ValueError("Invalid mode. Choose from 'one_hot', 'binary', or 'integer'")


def airy_disk(potential, resolution = 1.1):
    # make grid
    size_x = potential.shape[0]
    size_y = potential.shape[1]
    x = np.arange(size_x) - size_x//2 + 1
    y = np.arange(size_y) - size_y//2 + 1
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    pixel_size = potential.x.slope # Angstrom/pixel
    
    disk_radius = pixel_size / resolution * 2.5 # Airy disk radius in pixels
    # not sure why this 2.5 belonggs in here, but it works

    # Calculate the Airy pattern (PSF)
    with np.errstate(divide='ignore', invalid='ignore'):
        psf = (2 * sp.j1(disk_radius * rr) / (disk_radius * rr))**2
        psf[rr == 0] = 1  # Handling the division by zero at the center

    # Normalize the PSF
    psf /= np.sum(psf)
    
    dset = sidpy.Dataset.from_array(psf, name = 'Probe PSF')
    dset.data_type = 'image'
    dset.units = 'A.U.'
    dset.quantity = 'Probability'
    dset.set_dimension(0, sidpy.Dimension(pixel_size * np.arange(size_x),
                        name='x', units='Å', quantity='Length',dimension_type='spatial'))
    dset.set_dimension(1, sidpy.Dimension(pixel_size * np.arange(size_y),
                        name='y', units='Å', quantity='Length',dimension_type='spatial'))

    return dset

def get_probe(ab, potential, pixel_size=0.106):
    # pixel_size = potential.x.slope # Angstrom/pixel
    size_x, size_y = potential.shape

    probe, A_k, chi  = pt.get_probe(ab, size_x, size_y, verbose= True)

    dset = sidpy.Dataset.from_array(probe, name = 'Probe PSF')
    dset.data_type = 'image'
    dset.units = 'A.U.'
    dset.quantity = 'Probability'
    dset.set_dimension(0, sidpy.Dimension(pixel_size * np.arange(size_x),
                        name='x', units='Å', quantity='Length',dimension_type='spatial'))
    dset.set_dimension(1, sidpy.Dimension(pixel_size * np.arange(size_y),
                        name='y', units='Å', quantity='Length',dimension_type='spatial'))

    return dset


def convolve_kernel(potential, psf):
    # Convolve using FFT
    psf_shifted = da.fft.ifftshift(psf)
    image = da.fft.ifft2(da.fft.fft2(potential) * da.fft.fft2(psf_shifted))
    image = da.absolute(image)
    image = image - image.min()
    image = image / image.max()

    size_x, size_y = potential.shape
    pixel_size = potential.x.slope # Angstrom/pixel

    dset = potential.like_data(image)
    dset.units = 'A.U.'
    dset.quantity = 'Intensity'
    
    return dset


def poisson_noise(image, counts = 1e9):
    # Normalize the image
    image = image - image.min()
    image = image / image.sum()
    noisy_image = np.random.poisson(image * counts)

    noisy_image = noisy_image - noisy_image.min()
    noisy_image = noisy_image / noisy_image.max()
    noisy_image = image.like_data(noisy_image)

    return noisy_image


def lowfreq_noise(image, noise_level=0.1, freq_scale=0.1):
    size_x, size_y = image.shape

    noise = np.random.normal(0, noise_level, (size_x, size_y))
    noise_fft = np.fft.fft2(noise)

    # Create a frequency filter that emphasizes low frequencies
    x_freqs = np.fft.fftfreq(size_x)
    y_freqs = np.fft.fftfreq(size_y)
    freq_filter = np.outer(np.exp(-np.square(x_freqs) / (2 * freq_scale**2)),
                           np.exp(-np.square(y_freqs) / (2 * freq_scale**2)))

    # Apply the frequency filter to the noise in the frequency domain
    filtered_noise_fft = noise_fft * freq_filter
    low_freq_noise = np.fft.ifft2(filtered_noise_fft).real
    noisy_image = image + low_freq_noise
    noisy_image = image.like_data(noisy_image)

    return noisy_image


def grid_crop(image_master, crop_size=512, crop_glide=128):
    '''
    Slices an image into smaller, overlapping square crops.

    This function takes a larger image and divides it into smaller, overlapping square segments. 
    It's useful for processing large images in smaller batches, especially in machine learning applications 
    where input size is fixed.

    Parameters:
    - image_master: A NumPy array representing the image to be cropped. 
                    It should be a 2D array if the image is grayscale, or a 3D array for RGB images.
    - crop_size (int, optional): The size of each square crop. Default is 256 pixels.
    - crop_glide (int, optional): The stride or glide size for cropping. 
                                 Determines the overlap between consecutive crops. Default is 128 pixels.

    Returns:
    - cropped_ims: A NumPy array containing the cropped images. 
                   The array is 3D, where the first dimension represents the index of the crop, 
                   and the next two dimensions represent the height and width of the crops.

    Note:
    - The function assumes the input image is square. Non-square images might lead to unexpected results.
    - The return array is of type 'float16' to reduce memory usage, which might affect the precision of pixel values.
    '''

    n_crops = int((len(image_master) - crop_size)/crop_glide + 1)
    cropped_ims = np.zeros((n_crops,n_crops,crop_size,crop_size))

    for x in np.arange(n_crops):
        for y in np.arange(n_crops):
            xx,yy = int(x*crop_glide), int(y*crop_glide)
            cropped_ims[int(x),int(y)] = image_master[xx:xx+crop_size,yy:yy+crop_size]
    cropped_ims = cropped_ims.reshape((-1,crop_size,crop_size)).astype('float16')

    return cropped_ims


def resize_image(array, n, order = 3):
    """
    Resize a numpy array to n x n using interpolation.

    Parameters:
    array (numpy.ndarray): The input array.
    n (int): The size of the new square array.

    Returns:
    numpy.ndarray: The resized square array.
    """
    # Get the current shape of the array
    height, width = array.shape[-2:]

    # Calculate zoom factors
    zoom_factor = n / max(height, width)
    array = array.astype(np.float32)

    if len(array.shape) == 2:
        return zoom(array, [zoom_factor, zoom_factor], order = order)
    elif len(array.shape) == 3:
        return zoom(array, [1,zoom_factor, zoom_factor], order = order)


def shotgun_crop(image, crop_size=512, magnification_var = None, n_crops=10, seed=42, return_binary = False, roi = 'middle'):
    """
    Randomly crops a specified number of sub-images from a given image with variable magnification, supporting images with any number of channels.

    Parameters:
    image (numpy.ndarray): The input image as a NumPy array.
    crop_size (int, optional): The default size for each square crop. Defaults to 512.
    magnification_var (float, optional): The range of magnification variability as a fraction of the crop size. 
        If specified, each crop will be randomly sized within [crop_size * (1 - magnification_var), crop_size * (1 + magnification_var)]. Defaults to None.
    n_crops (int, optional): The number of crops to generate. Defaults to 10.
    seed (int, optional): Seed for the random number generator for reproducibility. Uses random package.

    Returns:
    numpy.ndarray: An array containing the cropped (and potentially resized) images as NumPy arrays.

    Important:
    If using this funciton on an image and mask together, make sure to use the same seed for both.
    """

    if return_binary == True:
        order = 0
    else:
        order = 3

    # Set seed for reproducibility
    # Seed should be a very large integer for good results
    crop_rng = np.random.default_rng(seed)

    # Get crop sizes for changing magnification later
    if magnification_var is not None:
        crop_sizes = crop_rng.integers(crop_size * ( 1 - magnification_var), crop_size * (1 + magnification_var), n_crops)
        crop_sizes = crop_sizes.astype(int)
    else:
        crop_sizes = np.full(n_crops, crop_size)

    # Randomly crop images (position and size)
    h, w = image.shape[-2:]
    crops = []
    for size in crop_sizes:
        if roi == 'middle':
            edge_cutoff = crop_size//4
            top = crop_rng.integers(edge_cutoff, h - size - edge_cutoff)
            left = crop_rng.integers(edge_cutoff, w - size - edge_cutoff)
        else:
            top = crop_rng.integers(0, h - size)
            left = crop_rng.integers(0, w - size)
        if len(image.shape) > 2:
            crop = image[:, top:top+size, left:left+size]
            crop = resize_image(crop, crop_size, order)
        else:
            crop = image[top:top+size, left:left+size]
            crop = resize_image(crop, crop_size, order)
        crops.append(crop)

    crops = np.array(crops)
    batch_crops = np.stack(crops, axis=0)
        
    return batch_crops
