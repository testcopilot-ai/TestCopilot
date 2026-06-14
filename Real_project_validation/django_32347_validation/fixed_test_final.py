from django import forms
from django.db import connection, models
from django.test import TransactionTestCase


class ChoiceModel32347(models.Model):
    name = models.CharField(max_length=80)

    class Meta:
        app_label = "testcopilot_generated"
        db_table = "tc_django_32347_choice"


class Ticket32347RegressionTests(TransactionTestCase):
    available_apps = []

    def create_table(self):
        with connection.schema_editor() as schema_editor:
            try:
                schema_editor.delete_model(ChoiceModel32347)
            except Exception:
                pass
            schema_editor.create_model(ChoiceModel32347)

    def drop_table(self):
        with connection.schema_editor() as schema_editor:
            try:
                schema_editor.delete_model(ChoiceModel32347)
            except Exception:
                pass

    def assert_invalid_choice_message_contains_value(self, submitted_value):
        self.create_table()
        try:
            field = forms.ModelChoiceField(
                queryset=ChoiceModel32347.objects.all(),
                error_messages={
                    "invalid_choice": '"%(value)s" is not one of the available choices.',
                },
            )
            with self.assertRaisesMessage(
                forms.ValidationError,
                f'"{submitted_value}" is not one of the available choices.',
            ):
                field.clean(submitted_value)
        finally:
            self.drop_table()

    def test_invalid_choice_text(self):
        self.assert_invalid_choice_message_contains_value("invalid")

    def test_invalid_choice_number_like_string(self):
        self.assert_invalid_choice_message_contains_value("12345")

    def test_invalid_choice_hyphenated_value(self):
        self.assert_invalid_choice_message_contains_value("missing-choice")

    def test_invalid_choice_mixed_case_value(self):
        self.assert_invalid_choice_message_contains_value("MissingChoice")

    def test_invalid_choice_symbol_value(self):
        self.assert_invalid_choice_message_contains_value("bad.choice")

    def test_invalid_choice_long_value(self):
        self.assert_invalid_choice_message_contains_value("ticket32347-invalid-value")