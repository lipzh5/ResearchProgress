# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import gdown

url_prefix = 'https://drive.google.com/uc?id='

def download_src():
	if not os.path.exists('ur5e/ur5e.urdf'):
		gdown.download(url=url_prefix+'1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc', output='ur5e.zip')
		gdown.download(url=url_prefix + '1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX', output='robotiq_2f_85.zip')
		gdown.download(url=url_prefix + '1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM', output='bowl.zip')


def download_with_id(download_id):
	gdown.download(url=url_prefix+download_id)

