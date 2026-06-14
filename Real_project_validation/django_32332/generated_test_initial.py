import uuid
from django.db import models
from django.test import TestCase


class CharPKParent(models.Model):
    id = models.CharField(primary_key=True, max_length=100)

    class Meta:
        app_label = 'test_char_pk'


class CharPKChild(models.Model):
    parent = models.ForeignKey(CharPKParent, on_delete=models.CASCADE)

    class Meta:
        app_label = 'test_char_pk'


class UUIDPKParent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    class Meta:
        app_label = 'test_uuid_pk'


class UUIDPKChild(models.Model):
    parent = models.ForeignKey(UUIDPKParent, on_delete=models.CASCADE)

    class Meta:
        app_label = 'test_uuid_pk'


class CharPKOneToOneChild(models.Model):
    parent = models.OneToOneField(CharPKParent, on_delete=models.CASCADE)

    class Meta:
        app_label = 'test_char_pk'


class CharPKChildCustomSave(models.Model):
    parent = models.ForeignKey(CharPKParent, on_delete=models.CASCADE)

    def save(self, *args, **kwargs):
        # Custom save that does some logic before calling super
        super().save(*args, **kwargs)

    class Meta:
        app_label = 'test_char_pk'


class Ticket32332Tests(TestCase):
    def test_char_pk_foreign_key_preserved_after_save(self):
        """
        Scenario 1: Saving a child model after saving a parent with a CharField
        primary key should preserve the ForeignKey relationship.
        """
        parent = CharPKParent.objects.create(id='parent-1')
        child = CharPKChild(parent=parent)
        child.save()
        self.assertEqual(child.parent_id, 'parent-1')
        # Refresh from DB to ensure it's persisted
        child.refresh_from_db()
        self.assertEqual(child.parent_id, 'parent-1')

    def test_uuid_pk_foreign_key_preserved_after_save(self):
        """
        Scenario 2: Saving a child model after saving a parent with a UUID
        primary key should preserve the ForeignKey relationship.
        """
        parent = UUIDPKParent.objects.create()
        child = UUIDPKChild(parent=parent)
        child.save()
        self.assertEqual(child.parent_id, parent.pk)
        child.refresh_from_db()
        self.assertEqual(child.parent_id, parent.pk)

    def test_char_pk_one_to_one_preserved_after_save(self):
        """
        Scenario 3: Saving a child model after saving a parent with a CharField
        primary key using OneToOneField should preserve the relationship.
        """
        parent = CharPKParent.objects.create(id='parent-2')
        child = CharPKOneToOneChild(parent=parent)
        child.save()
        self.assertEqual(child.parent_id, 'parent-2')
        child.refresh_from_db()
        self.assertEqual(child.parent_id, 'parent-2')

    def test_char_pk_foreign_key_preserved_after_multiple_saves(self):
        """
        Scenario 4: Saving a child model multiple times after saving a parent
        with a CharField primary key should preserve the ForeignKey relationship.
        """
        parent = CharPKParent.objects.create(id='parent-3')
        child = CharPKChild(parent=parent)
        child.save()
        child.save()  # Second save without changes
        self.assertEqual(child.parent_id, 'parent-3')
        child.refresh_from_db()
        self.assertEqual(child.parent_id, 'parent-3')

    def test_char_pk_foreign_key_preserved_with_custom_save(self):
        """
        Scenario 5: Saving a child model with a custom save method after saving
        a parent with a CharField primary key should preserve the ForeignKey
        relationship.
        """
        parent = CharPKParent.objects.create(id='parent-4')
        child = CharPKChildCustomSave(parent=parent)
        child.save()
        self.assertEqual(child.parent_id, 'parent-4')
        child.refresh_from_db()
        self.assertEqual(child.parent_id, 'parent-4')

    def test_char_pk_foreign_key_preserved_with_retrieved_parent(self):
        """
        Scenario 6: Saving a child model after retrieving the parent from the
        database should preserve the ForeignKey relationship.
        """
        CharPKParent.objects.create(id='parent-5')
        parent = CharPKParent.objects.get(id='parent-5')
        child = CharPKChild(parent=parent)
        child.save()
        self.assertEqual(child.parent_id, 'parent-5')
        child.refresh_from_db()
        self.assertEqual(child.parent_id, 'parent-5')

    def test_char_pk_foreign_key_preserved_via_model_form(self):
        """
        Scenario 7: Saving a child model via a model form after saving a parent
        with a CharField primary key should preserve the ForeignKey relationship.
        """
        from django import forms

        class CharPKChildForm(forms.ModelForm):
            class Meta:
                model = CharPKChild
                fields = ['parent']

        parent = CharPKParent.objects.create(id='parent-6')
        form = CharPKChildForm(data={'parent': parent.pk})
        self.assertTrue(form.is_valid())
        child = form.save()
        self.assertEqual(child.parent_id, 'parent-6')
        child.refresh_from_db()
        self.assertEqual(child.parent_id, 'parent-6')