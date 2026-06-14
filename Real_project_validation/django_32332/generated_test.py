from django.db import connection, models
from django.test import TransactionTestCase


class Product32332(models.Model):
    sku = models.CharField(primary_key=True, max_length=80)

    class Meta:
        app_label = "testcopilot_generated"
        db_table = "tc_django_32332_product"


class Order32332(models.Model):
    product = models.ForeignKey(Product32332, on_delete=models.CASCADE)

    class Meta:
        app_label = "testcopilot_generated"
        db_table = "tc_django_32332_order"


class Ticket32332RegressionTests(TransactionTestCase):
    available_apps = []

    def create_tables(self):
        with connection.schema_editor() as schema_editor:
            try:
                schema_editor.delete_model(Order32332)
            except Exception:
                pass
            try:
                schema_editor.delete_model(Product32332)
            except Exception:
                pass
            schema_editor.create_model(Product32332)
            schema_editor.create_model(Order32332)

    def drop_tables(self):
        with connection.schema_editor() as schema_editor:
            try:
                schema_editor.delete_model(Order32332)
            except Exception:
                pass
            try:
                schema_editor.delete_model(Product32332)
            except Exception:
                pass

    def assert_child_fk_tracks_parent_pk_assigned_after_relation(self, sku):
        self.create_tables()
        try:
            parent = Product32332()
            child = Order32332(product=parent)

            child.product.sku = sku
            parent.save()
            child.save()
            child.refresh_from_db()

            self.assertEqual(child.product_id, sku)
            self.assertEqual(child.product, parent)
        finally:
            self.drop_tables()

    def test_char_pk_alpha(self):
        self.assert_child_fk_tracks_parent_pk_assigned_after_relation("alpha")

    def test_char_pk_hyphen(self):
        self.assert_child_fk_tracks_parent_pk_assigned_after_relation("sku-001")

    def test_char_pk_mixed_case(self):
        self.assert_child_fk_tracks_parent_pk_assigned_after_relation("SkuMixedCase")

    def test_char_pk_alphanumeric(self):
        self.assert_child_fk_tracks_parent_pk_assigned_after_relation("A100B200")

    def test_char_pk_underscore(self):
        self.assert_child_fk_tracks_parent_pk_assigned_after_relation("product_key_05")

    def test_char_pk_long_value(self):
        self.assert_child_fk_tracks_parent_pk_assigned_after_relation("ticket32332")
