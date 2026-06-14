1. Scenario: ModelChoiceField with invalid primary key value
   Condition: A ModelChoiceField is bound with a value that does not correspond to any existing object's primary key (e.g., a non-existent integer ID).
   Expected behavior: The field should raise a ValidationError with code 'invalid_choice', and the error should include the invalid value in its params dictionary under the key 'value'.
   Why this exposes the issue: The buggy version does not pass the invalid submitted value to the ValidationError's params, so the error message cannot display the actual invalid value.

2. Scenario: ModelChoiceField with invalid string value
   Condition: A ModelChoiceField is bound with a string value that is not a valid choice (e.g., a random string when the field expects integer primary keys).
   Expected behavior: The field should raise a ValidationError with code 'invalid_choice', and the error's params should contain the submitted string value.
   Why this exposes the issue: The buggy version fails to include the submitted value in the error parameters, making it impossible to identify which value was rejected.

3. Scenario: ModelChoiceField with empty string value
   Condition: A ModelChoiceField with required=False is bound with an empty string as the value.
   Expected behavior: If the empty string is not a valid choice, the field should raise a ValidationError with code 'invalid_choice', and the error's params should include the empty string value.
   Why this exposes the issue: The buggy version does not pass the empty string to the error params, so the error message cannot show the rejected value.

4. Scenario: ModelChoiceField with value that is a valid object but not in queryset
   Condition: A ModelChoiceField is bound with a primary key value that exists in the database but is not included in the field's queryset (e.g., due to limit_choices_to).
   Expected behavior: The field should raise a ValidationError with code 'invalid_choice', and the error's params should contain the submitted value.
   Why this exposes the issue: The buggy version does not include the invalid value in the error params, even though the value is technically a valid object but not a valid choice for this field.

5. Scenario: ModelChoiceField with value that is not coercible to the expected type
   Condition: A ModelChoiceField expects integer primary keys, but is bound with a value that cannot be converted to an integer (e.g., a very long string or special characters).
   Expected behavior: The field should raise a ValidationError with code 'invalid_choice', and the error's params should include the original submitted value.
   Why this exposes the issue: The buggy version does not preserve the submitted value in the error params, so the error message cannot display the problematic input.

6. Scenario: ModelChoiceField with value that is a valid UUID but not in queryset
   Condition: A ModelChoiceField uses UUID primary keys, and is bound with a valid UUID string that does not match any existing object.
   Expected behavior: The field should raise a ValidationError with code 'invalid_choice', and the error's params should contain the submitted UUID string.
   Why this exposes the issue: The buggy version does not include the invalid UUID value in the error params, making it difficult to identify which UUID was rejected.

7. Scenario: ModelChoiceField with value that is a valid object but from a different model
   Condition: A ModelChoiceField is bound with a primary key value that exists in the database but belongs to a different model (e.g., using an Author's ID in a field that expects Book IDs).
   Expected behavior: The field should raise a ValidationError with code 'invalid_choice', and the error's params should contain the submitted value.
   Why this exposes the issue: The buggy version does not pass the invalid value to the error params, so the error message cannot show which value was invalid.