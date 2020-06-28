from PIL import Image

lena = Image.open('./01-Data/lena.png')
lena_modified = Image.open('./01-Data/lena_modified.png')

w, h = lena.size
for i in range(h):
    for j in range(w):
        if lena.getpixel((i, j)) == lena_modified.getpixel((i, j)):
            lena_modified.putpixel((i, j), 255)

lena_modified.show()
lena_modified.save('./02-Output/Q2_ans_two.png')
