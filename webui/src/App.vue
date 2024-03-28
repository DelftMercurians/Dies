<template>
  <v-app>
    <v-main>
      <v-row no-gutters class="fill-height">
        <!-- Player sidebar -->
        <v-col cols="3">
          <BotDetails></BotDetails>
        </v-col>

        <!-- Field -->
        <v-col
          cols="6"
          class="d-flex align-center"
          style="background-color: rgb(31, 31, 31)"
        >
          <Field></Field>
        </v-col>

        <!-- ? Some sidebar that does nothing yet -->
        <v-col cols="3"></v-col>
      </v-row>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
// ---------------- COMPONENT IMPORTS ----------------
import Field from "./components/Field.vue";
import BotDetails from "./components/BotDetails.vue";
</script>
