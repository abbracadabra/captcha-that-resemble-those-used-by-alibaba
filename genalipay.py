import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageFilter
import PIL.Image as Image
import PIL
import PIL.ImageOps
import os
import uuid
import PIL.ImageFilter as ImageFilter
import colorsys

dir = os.path.dirname(os.path.realpath(__file__))
savepath = os.path.join(dir,'pice')
fontpath = os.path.join(dir,'fonts')

def perspective(im,widthscale,heightscale,horizontaloblique,verticaloblique,shiftright,shiftdown,MagnifyX,MagnifyY,size=None):
    """

    :param im:
    :param widthscale: by which width is multiplied
    :param heightscale: by which height is multiplied
    :param horizontaloblique: Oblique distortion.Think about transformation between normal font to oblique font
    :param verticaloblique: Oblique distortion vertically.
    :param shiftright: Normal shift,rightward.No distortion applied
    :param shiftdown: Normal shift,downward
    :param MagnifyX: Magnify image.Regions further along X axis will be magnified more.
    :param MagnifyY: Magnify image.Regions further along Y axis will be magnified more.
    :return:
    """
    if size==None:
        size = im.size
    #Data is a 8-tuple (a, b, c, d, e, f, g, h)
    #For each pixel (x, y) in the output image, the new value is taken from a position (a x + b y + c)/(g x + h y + 1), (d x + e y + f)/(g x + h y + 1) in the input image
    im = im.transform(size, Image.PERSPECTIVE, (1/widthscale, -horizontaloblique, -shiftright, -verticaloblique, 1/heightscale, -shiftdown, MagnifyX, MagnifyY), fillcolor=(255, 255, 255))
    im = PIL.ImageOps.invert(im)
    im = im.crop(im.getbbox())
    im = PIL.ImageOps.invert(im)
    return im

def affine(im,widthscale,heightscale,horizontaloblique,verticaloblique,shiftright,shiftdown,size=None):
    """

    :param im:
    :param widthscale:
    :param heightscale:
    :param horizontaloblique:
    :param verticaloblique:
    :param shiftright:
    :param shiftdown:
    :param size:
    :return:
    """

    if size==None:
        size=im.size
    else:
        im_resize = Image.new('RGB', size, (255, 255, 255))
        im_resize.paste(im, ((size[0] - im.size[0]) // 2, (size[1] - im.size[1]) // 2, (size[0] + im.size[0]) // 2,
        (size[1] + im.size[1]) // 2))
        im = im_resize
    #Data is a 6-tuple (a, b, c, d, e, f)
    # For each pixel (x, y) in the output image, the new value is taken from a position (a x + b y + c, d x + e y + f) in the input image
    im = im.transform(size, Image.AFFINE, (1/widthscale, -horizontaloblique, -shiftright, -verticaloblique, 1/heightscale, -shiftdown), fillcolor=(255, 255, 255))
    im = PIL.ImageOps.invert(im)
    im = im.crop(im.getbbox())
    im = PIL.ImageOps.invert(im)
    return im

def zoom(im,scale,size=None):
    if size==None:
        size = im.size
    of = (1-1/scale)/2
    #Data is 4-tuple box coord
    im = im.transform(size, Image.EXTENT, (im.size[0]*of,im.size[1]*of,im.size[0]*(1-of),im.size[1]*(1-of)), fillcolor=(255, 255, 255))
    return im

def radial(im_pil,alpha,centre=(0.5,0.5),fill=(255,255,255),size=None):
    """

    Fitzgibbon, 2001  https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion/6227310#6227310
    barrel distortion: rd = ru(1- alpha * ru^2)  ru = rd/(1- alpha * rd^2)
    """
    if size==None:
        size=im_pil.size
    else:
        im_pil_resize = Image.new('RGB', size, (255, 255, 255))
        im_pil_resize.paste(im_pil, ((size[0] - im_pil.size[0]) // 2, (size[1] - im_pil.size[1]) // 2, (size[0] + im_pil.size[0]) // 2,
        (size[1] + im_pil.size[1]) // 2))
        im_pil = im_pil_resize
        centre = ((im_pil.size[0] * centre[0] + (size[0] - im_pil.size[0]) // 2) / size[0],
                  (im_pil.size[1] * centre[1] + (size[1] - im_pil.size[1]) // 2) / size[1])

    im = np.array(im_pil)
    im_pil.close()
    w = im.shape[1]
    h = im.shape[0]
    index_centre = ((h-1)*centre[1],(w-1)*centre[0])
    indices_x,indices_y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dst_indices = np.stack((indices_x, indices_y), axis=-1)
    ru_inverse_rd = 1/(1 - np.sum((dst_indices-index_centre)**2,axis=-1,keepdims=True)*alpha)
    map_indices = np.int16(index_centre + (dst_indices-index_centre)*ru_inverse_rd)
    def interpolate(i_index,j_index):
        if (i_index >= 0 and i_index < h and j_index >= 0 and j_index < w):
            mm = [im[i_index,j_index]]
            try:
                mm.append(im[i_index+1,j_index])
            except:
                pass
            try:
                mm.append(im[i_index,j_index+1])
            except:
                pass
            return np.max(mm,axis=0)
        else:
            return fill

    out = np.array([interpolate(i_index,j_index) for i_index,j_index in np.reshape(map_indices,(-1,2))])
    out = np.reshape(out,im.shape)
    im = Image.fromarray(np.uint8(out))
    im = PIL.ImageOps.invert(im)
    im = im.crop(im.getbbox())
    im = PIL.ImageOps.invert(im)
    return im

def mustache():
    pass

def randomdistort(im):
    im = PIL.ImageOps.invert(im)
    im = im.crop(im.getbbox())
    im = PIL.ImageOps.invert(im)
    im = im.rotate(np.random.uniform(-15,5))
    rt_bg = Image.new('RGB',im.size,(255, 255, 255))
    rt_bg.paste(im,(0,0),mask = im.split()[2])
    im = rt_bg
    im = PIL.ImageOps.invert(im)
    im = im.crop(im.getbbox())
    im = PIL.ImageOps.invert(im)

    #affine
    im = affine(im,widthscale=np.random.uniform(0.7,1.3),
                heightscale=np.random.uniform(0.7,1.3),
                horizontaloblique=np.random.uniform(-0.7,0),
                verticaloblique=np.random.uniform(-0.4,0.4),
                shiftright=0,
                shiftdown=0,size=(90,90))

    # flip
    lr = np.random.choice([0, 1])
    tb = np.random.choice([0, 1])
    if lr==1:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if tb==1:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

    #perspective
    im = perspective(im,
                widthscale=1,
                heightscale=1,
                horizontaloblique=0,
                verticaloblique=0,
                #shiftright=np.random.randint(0,int(w*0.5))*np.random.choice([1,-1]),
                shiftright=0,
                shiftdown=0,
                MagnifyX=np.random.uniform(-0.003,0.003),
                MagnifyY=np.random.uniform(-0.003,0.003),
                size=(90,90))

    #radial
    im = radial(im, np.random.uniform(-0.000005,0.000005), centre=(np.random.rand(),np.random.uniform(0.3,0.7)), fill=(255, 255, 255),size=(90,90))
    
    if lr==1:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if tb==1:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

    return im

chars = []
for i in range(10):
    chars.append(chr(i+48))
for i in range(26):
    chars.append(chr(i + 65))

fonts = [f for f in os.listdir(fontpath)]

def makeone():
    im = PIL.Image.new('RGB', (200, 60), (255, 255, 255))
    draw = PIL.ImageDraw.Draw(im)
    cms = []
    fourc = ''

    for i in range(4):
        c = np.random.choice(chars)
        fn = np.random.choice(fonts)
        fp = os.path.join(fontpath, fn)
        font = PIL.ImageFont.truetype(font=fp, size=np.random.randint(42,54))
        cw, ch = draw.textsize(c, font=font)
        imc = PIL.Image.new('RGB', (cw, ch), (255, 255, 255))
        cdraw = PIL.ImageDraw.Draw(imc)
        cdraw.text((0, 0), c, font=font, fill=(0, 0, 255))
        imc = randomdistort(imc)
        cms.append(imc)
        fourc+=c

    g1, g2, g3 = np.int16(np.random.randn(3) * 4 - 2)
    overw = cms[0].size[0] + cms[1].size[0] + cms[2].size[0] + cms[3].size[0] + g1 + g2 + g3
    left = np.random.randint(200 - overw)

    im.paste(cms[0], (left, np.random.randint(0, max(60 - cms[0].size[1], 1))),
             mask=cms[0].split()[0].point(lambda x: (x < 255) * 255))
    im.paste(cms[1], (left + cms[0].size[0] + g1, np.random.randint(0, max(60 - cms[1].size[1], 1))),
             mask=cms[1].split()[0].point(lambda x: (x < 255) * 255))
    im.paste(cms[2],
             (left + cms[0].size[0] + g1 + cms[1].size[0] + g2, np.random.randint(0, max(60 - cms[2].size[1], 1))),
             mask=cms[2].split()[0].point(lambda x: (x < 255) * 255))
    im.paste(cms[3], (left + cms[0].size[0] + g1 + cms[1].size[0] + g2 + cms[2].size[0] + g3,
                      np.random.randint(0, max(60 - cms[3].size[1], 1))),
             mask=cms[3].split()[0].point(lambda x: (x < 255) * 255))
    im = im.resize((100, 30))
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    im.save(os.path.join(savepath,
                         fourc + "_" + "".join(x for x in fn if x.isalnum()) + "_" + str(uuid.uuid4()) + ".jpg"), 'JPEG')


for i in range(100000):
    try:
        makeone()
    except:
        pass


    

