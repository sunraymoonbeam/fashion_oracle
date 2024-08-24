import pytest
import tensorflow as tf

from src.modelling.model_factory import ModelFactory


# you can use the scope parameter in the pytest.fixture decorator to control the fixture's lifespan.
# By setting the scope to "module", the fixture will be created once per module, and the same instance will be used
# across all tests in that module.
# This will help in persisting the ModelFactory instance and avoid creating multiple models unnecessarily.


@pytest.fixture(scope="module")  # This fixture will be created once per module
def model_factory():
    return ModelFactory()


@pytest.fixture(scope="module")
def base_model():
    return tf.keras.applications.VGG16(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )


def test_list_available_models(model_factory):
    expected_models = [
        "VGG16",
        "InceptionV3",
        "ResNet50",
        "MobileNetV2",
        "EfficientNetB0",
        "DenseNet121",
        "NASNetMobile",
    ]
    assert model_factory.list_available_models() == expected_models


def test_create_base_model(model_factory):
    base_model = model_factory.create_base_model(
        base_model_name="VGG16",
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    assert isinstance(base_model, tf.keras.Model)
    assert base_model.input_shape == (None, 224, 224, 3)


def test_create_base_model_invalid_name(model_factory):
    with pytest.raises(
        ValueError, match="Model InvalidModel not available in the factory."
    ):
        model_factory.create_base_model(
            base_model_name="InvalidModel",
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
        )


def test_build_model(model_factory):
    model = model_factory.build_model(
        base_model_name="VGG16", input_shape=(224, 224, 3), num_classes=10
    )
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 10)
