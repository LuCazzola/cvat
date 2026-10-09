// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Row, Col } from 'antd/lib/grid';
import Text from 'antd/lib/typography/Text';
import Slider from 'antd/lib/slider';

import { CombinedState } from 'reducers';
import { enableImageFilter, disableImageFilter } from 'actions/settings-actions';
import { ImageFilterAlias, hasFilter } from 'utils/image-processing';
import Highlights, { HighlightsOptions } from 'utils/highlights';

const OFF: HighlightsOptions = { knee: 0.7, strength: 0 };

export default function HighlightsFilter(): JSX.Element {
    const dispatch = useDispatch();
    const filters = useSelector((state: CombinedState) => state.settings.imageFilters);
    const filter = hasFilter(filters, ImageFilterAlias.HIGHLIGHTS);
    const options = filter ? (filter.modifier as Highlights).options : OFF;

    const update = (changed: Partial<HighlightsOptions>): void => {
        const next = { ...options, ...changed };
        if (next.strength === 0) {
            if (filter) dispatch(disableImageFilter(ImageFilterAlias.HIGHLIGHTS));
        } else if (filter) {
            dispatch(enableImageFilter(filter, next));
        } else {
            dispatch(enableImageFilter({ modifier: new Highlights(next), alias: ImageFilterAlias.HIGHLIGHTS }));
        }
    };

    return (
        <div className='cvat-image-setups-filters cvat-image-setups-highlights'>
            <Row>
                <Col span={6}>
                    <Text className='cvat-text-color'> Highlights </Text>
                </Col>
                <Col span={12}>
                    <Slider
                        min={0}
                        max={9}
                        step={0.1}
                        value={options.strength}
                        onChange={(strength: number) => update({ strength })}
                    />
                </Col>
            </Row>
            <Row>
                <Col span={6}>
                    <Text className='cvat-text-color'> from </Text>
                </Col>
                <Col span={12}>
                    <Slider
                        min={0.3}
                        max={0.95}
                        step={0.01}
                        value={options.knee}
                        disabled={!filter}
                        onChange={(knee: number) => update({ knee })}
                    />
                </Col>
            </Row>
        </div>
    );
}
