import uuid
from django import forms
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.forms.models import ModelChoiceField
from django.db import models


class TestModel(models.Model):
    name = models.CharField(max_length=100)

    class Meta:
        app_label = 'test_model_choice_field'


class OtherModel(models.Model):
    name = models.CharField(max_length=100)

    class Meta:
        app_label = 'test_model_choice_field'


class UUIDTestModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)

    class Meta:
        app_label = 'test_model_choice_field'


class TestModelChoiceFieldInvalidChoiceValue(TestCase):
    def test_invalid_primary_key_value(self):
        """
        Scenario 1: ModelChoiceField with invalid primary key value.
        When a non-existent integer ID is submitted, the ValidationError
        should include the invalid value in its params.
        """
        field = ModelChoiceField(queryset=TestModel.objects.none())
        with self.assertRaises(ValidationError) as cm:
            field.clean(999)
        self.assertEqual(cm.exception.code, 'invalid_choice')
        self.assertIn('value', cm.exception.params)
        self.assertEqual(cm.exception.params['value'], 999)

    def test_invalid_string_value(self):
        """
        Scenario 2: ModelChoiceField with invalid string value.
        When a random string is submitted, the ValidationError
        should include the submitted string value in its params.
        """
        field = ModelChoiceField(queryset=TestModel.objects.none())
        with self.assertRaises(ValidationError) as cm:
            field.clean('invalid_string')
        self.assertEqual(cm.exception.code, 'invalid_choice')
        self.assertIn('value', cm.exception.params)
        self.assertEqual(cm.exception.params['value'], 'invalid_string')

    def test_empty_string_value(self):
        """
        Scenario 3: ModelChoiceField with empty string value.
        When an empty string is submitted and the field is not required,
        the ValidationError should include the empty string in its params.
        """
        field = ModelChoiceField(queryset=TestModel.objects.none(), required=False)
        with self.assertRaises(ValidationError) as cm:
            field.clean('')
        self.assertEqual(cm.exception.code, 'invalid_choice')
        self.assertIn('value', cm.exception.params)
        self.assertEqual(cm.exception.params['value'], '')

    def test_valid_object_not_in_queryset(self):
        """
        Scenario 4: ModelChoiceField with value that is a valid object
        but not in queryset (e.g., due to limit_choices_to).
        The ValidationError should include the submitted value in its params.
        """
        obj = TestModel.objects.create(name='test')
        field = ModelChoiceField(queryset=TestModel.objects.none())
        with self.assertRaises(ValidationError) as cm:
            field.clean(obj.pk)
        self.assertEqual(cm.exception.code, 'invalid_choice')
        self.assertIn('value', cm.exception.params)
        self.assertEqual(cm.exception.params['value'], obj.pk)

    def test_non_coercible_value(self):
        """
        Scenario 5: ModelChoiceField with value that is not coercible
        to the expected type (e.g., a very long string when integer expected).
        The ValidationError should include the original submitted value.
        """
        field = ModelChoiceField(queryset=TestModel.objects.none())
        with self.assertRaises(ValidationError) as cm:
            field.clean('a' * 1000)
        self.assertEqual(cm.exception.code, 'invalid_choice')
        self.assertIn('value', cm.exception.params)
        self.assertEqual(cm.exception.params['value'], 'a' * 1000)

    def test_valid_uuid_not_in_queryset(self):
        """
        Scenario 6: ModelChoiceField with value that is a valid UUID
        but not in queryset. The ValidationError should include the
        submitted UUID string in its params.
        """
        field = ModelChoiceField(queryset=UUIDTestModel.objects.none())
        test_uuid = uuid.uuid4()
        with self.assertRaises(ValidationError) as cm:
            field.clean(test_uuid)
        self.assertEqual(cm.exception.code, 'invalid_choice')
        self.assertIn('value', cm.exception.params)
        self.assertEqual(cm.exception.params['value'], test_uuid)

    def test_value_from_different_model(self):
        """
        Scenario 7: ModelChoiceField with value that is a valid object
        but from a different model. The ValidationError should include
        the submitted value in its params.
        """
        other_obj = OtherModel.objects.create(name='other')
        field = ModelChoiceField(queryset=TestModel.objects.none())
        with self.assertRaises(ValidationError) as cm:
            field.clean(other_obj.pk)
        self.assertEqual(cm.exception.code, 'invalid_choice')
        self.assertIn('value', cm.exception.params)
        self.assertEqual(cm.exception.params['value'], other_obj.pk)