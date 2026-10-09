// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { ImageProcessing } from 'cvat-core-wrapper';

export enum ImageFilterAlias {
    HISTOGRAM_EQUALIZATION = 'opencv.histogramEqualization',
    ENHANCEMENT = 'opencv.enhancement',
    GAMMA_CORRECTION = 'fabric.gammaCorrection',
}

export interface ImageFilter {
    modifier: ImageProcessing,
    alias: ImageFilterAlias
}

export function hasFilter(filters: ImageFilter[], alias: ImageFilterAlias): ImageFilter | null {
    const index = filters.findIndex((imageFilter) => imageFilter.alias === alias);
    if (index !== -1) {
        return filters[index];
    }
    return null;
}

// RedScrap image enhancement methods (cvat-core opencv/enhancement.ts): sliders and production defaults
export const ENHANCEMENTS: Record<string, { label: string, min: number, max: number, step: number, value: number }[]> = {
    clahe_luminance: [{ label: 'clipLimit', min: 0.5, max: 10, step: 0.5, value: 2 }],
    clahe_channels: [{ label: 'clipLimit', min: 0.5, max: 10, step: 0.5, value: 2 }],
    he_luminance: [],
    he_channels: [],
    rek: [{ label: 'p', min: 0.5, max: 5, step: 0.1, value: 2 }, { label: 'alpha', min: -1, max: 5, step: 0.1, value: -1 }],
    riesz: [
        { label: 'alpha', min: 0.1, max: 1, step: 0.05, value: 0.4 },
        { label: 'gainHigh', min: 0.5, max: 3, step: 0.1, value: 1.6 },
        { label: 'gainLow', min: 0.1, max: 1.5, step: 0.1, value: 0.6 },
    ],
    riesz_luminance_only: [
        { label: 'alpha', min: 0.1, max: 1, step: 0.05, value: 0.4 },
        { label: 'gainHigh', min: 0.5, max: 3, step: 0.1, value: 1.6 },
        { label: 'gainLow', min: 0.1, max: 1.5, step: 0.1, value: 0.6 },
    ],
};
