// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { BaseImageFilter, SerializedImageFilter } from './image-processing';

export interface ClaheOptions {
    clipLimit: number;
}

// CLAHE on the L channel of Lab, like clahe_luminance in RedScrap's image_enhancement (8x8 tiles, clip 2 there)
export default class ClaheImplementation extends BaseImageFilter {
    private cv: any;
    public clipLimit: number;

    constructor(cv: any, options: ClaheOptions) {
        super();
        this.cv = cv;
        this.clipLimit = options.clipLimit;
    }

    public configure(options: object): void {
        this.clipLimit = (options as ClaheOptions).clipLimit;
    }

    public toJSON(): SerializedImageFilter {
        return { alias: 'opencv.clahe', params: { clipLimit: this.clipLimit } };
    }

    public processImage(src: ImageData, frameNumber: number): ImageData {
        const { cv } = this;
        const mats: any[] = [];
        const mat = (): any => { const m = new cv.Mat(); mats.push(m); return m; };
        const channels = new cv.MatVector();
        const clahe = new cv.CLAHE(this.clipLimit, new cv.Size(8, 8));
        try {
            this.currentProcessedImage = frameNumber;
            const rgba = cv.matFromImageData(src);
            mats.push(rgba);
            const rgb = mat();
            const lab = mat();
            cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB, 0);
            cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab, 0);
            cv.split(lab, channels);
            const L = channels.get(0);
            mats.push(L);
            const equalized = mat();
            clahe.apply(L, equalized);
            channels.set(0, equalized);
            cv.merge(channels, lab);
            cv.cvtColor(lab, rgb, cv.COLOR_Lab2RGB, 0);
            const out = mat();
            cv.cvtColor(rgb, out, cv.COLOR_RGB2RGBA, 0);
            return new ImageData(new Uint8ClampedArray(out.data), src.width, src.height);
        } finally {
            mats.forEach((m) => m.delete());
            channels.delete();
            clahe.delete();
        }
    }
}
