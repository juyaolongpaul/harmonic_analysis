\version "2.10.33"

\header {
  title = "226 - Herr Jesu Christ, du hast bereit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g8 a8 bes4 a8 g8 f4 
    bes4 a4 bes4 bes4 
    a4 g4 c4 bes4 
    a2 g4 } d'4 
  bes4 c4 d4 c4   %5
  bes4 bes4 a4 bes4 
  bes4 a4 bes4 a4 
  g4 g4 fis4 bes4 
  a4 g4 c4 bes4 
  a2 g4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    ees4 f8 g8 f4 c4 
    f4 f4 f4 g4~ 
    g8 fis8 g4. a4 g8 
    g8 fis16 e16 fis4 d4 } a'4~ 
  a8 g4 f8 f4 f8. ees16   %5
  d8 e16 fis16 g4 fis4 g4 
  f4. ees8 d8 g4 fis8 
  g8 fis8 g8 g,8 d'4 g,8 g'8~ 
  g8 fis8 g8 f8 ees4 d4 
  ees8 c8 a8 d16 c16 b4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    bes8 c8 d4 c8 bes8 a4 
    bes8 d8 c16 d16 ees4 d8 d4 
    ees8 d16 c16 bes8 b8 c8 ees8 d4 
    ees8 c8 a8 d16 c16 bes4 } d4 
  d4 c4. bes4 a8   %5
  bes8. c16 d4 d4 d4 
  d4 c4 bes8 d8 ees8 d8~ 
  d4. c8 a4 d4 
  d8. c16 bes8 a8 g8 a4 g8~ 
  g8 fis16 e16 fis4 d   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g8  f8 
    ees4 d8 ees8 f4. ees8 
    d8 bes8 f'4 bes,4 g4 
    c8 d8 ees8 d8 ees8 c8 g8 bes8 
    c8 a8 d4 g,4 } fis'4 
  g4 a4 bes4 f4   %5
  g8. a16 bes8 g8 d4 g4 
  d8 ees8 f8 fis8 g8 bes,8 c8 d8 
  ees2 d4 g,4 
  d'4 ees4. fis,8 g8 bes8 
  c8 a8 d4 g,   %10
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}