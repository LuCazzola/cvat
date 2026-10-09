// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { AnyAction } from 'redux';
import notification from 'antd/lib/notification';

import {
    changeBrightnessLevel, changeContrastLevel, changeSaturationLevel, enableImageFilter, resetImageFilters,
} from 'actions/settings-actions';
import { CombinedState } from 'reducers';
import { EnhancementOptions } from 'cvat-core/src/opencv/enhancement';
import GammaCorrection from 'utils/fabric-wrapper/gamma-correction';
import { ImageFilterAlias, hasFilter } from 'utils/image-processing';
import openCVWrapper from 'utils/opencv-wrapper/opencv-wrapper';

// An annotator's colour settings, saved on purpose in this browser under their CVAT username and applied whenever
// they open a job. CVAT's own clientSettings keep the last values instead, and are shared by every account.
interface ColorPreset {
    brightness: number;
    contrast: number;
    saturation: number;
    gamma: number;
    enhancement: EnhancementOptions | null;
}

const storageKey = (username: string): string => `redscrap.colorPreset.${username}`;

export function saveColorPreset(username: string, settings: CombinedState['settings']): void {
    const { brightnessLevel, contrastLevel, saturationLevel } = settings.player;
    const gamma = hasFilter(settings.imageFilters, ImageFilterAlias.GAMMA_CORRECTION);
    const enhancement = hasFilter(settings.imageFilters, ImageFilterAlias.ENHANCEMENT);
    const preset: ColorPreset = {
        brightness: brightnessLevel,
        contrast: contrastLevel,
        saturation: saturationLevel,
        gamma: gamma ? (gamma.modifier as GammaCorrection).gamma : 1,
        enhancement: enhancement ? {
            method: (enhancement.modifier as any).method,
            params: (enhancement.modifier as any).params,
        } : null,
    };
    localStorage.setItem(storageKey(username), JSON.stringify(preset));
}

export async function applyColorPreset(username: string, dispatch: (action: AnyAction) => void): Promise<void> {
    try {
        const saved = localStorage.getItem(storageKey(username));
        if (!saved) return;
        const preset: ColorPreset = JSON.parse(saved);
        dispatch(changeBrightnessLevel(preset.brightness));
        dispatch(changeContrastLevel(preset.contrast));
        dispatch(changeSaturationLevel(preset.saturation));
        dispatch(resetImageFilters());
        // pixel filters run in the order they are enabled: the enhancement first, gamma on its result
        if (preset.enhancement) {
            await openCVWrapper.initialize(() => {});
            dispatch(enableImageFilter({
                modifier: openCVWrapper.imgproc.enhancement(preset.enhancement),
                alias: ImageFilterAlias.ENHANCEMENT,
            }));
        }
        if (preset.gamma !== 1) {
            const { gamma } = preset;
            dispatch(enableImageFilter({
                modifier: new GammaCorrection({ gamma: [gamma, gamma, gamma] }),
                alias: ImageFilterAlias.GAMMA_CORRECTION,
            }));
        }
    } catch (error: unknown) {
        notification.error({
            message: 'Could not apply your saved colour settings',
            description: error instanceof Error ? error.message : String(error),
        });
    }
}
