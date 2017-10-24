# MAIN.py
#
# Quantization of an image : Uses the k-means algorithm to color-quantize the image to only k colors.
#
# Autor: Elodie Couturier
# ######################################################################################################################

from filters import *
import math

check_file = True
while check_file:
	try:
		print('\nEnter the path to the image you want to process:')
		image_path = input()
		im = FILTER(image_path)
		check_file = False
	except FileNotFoundError:
		print('No such file.')

k = -1
while k <= 1:
	print('Enter the number of colors you want your image to have.\nPlease enter a k > 1')
	k = int(input())

max_iter = -2
while (max_iter != 0) and (max_iter < -1):
	print('How many iterations do you want the k-means algorithm to do before it stops ? (if not converging)\n'
		  '(enter "-1" for an infinity of iterations)')
	max_iter = int(input())

if max_iter == -1:
	max_iter = math.inf

print('\nQuantization...')
im.quantization(k, max_iter)
print('Done!\nImage saved as ' + im.savename)

im.im.close()
