\version "2.10.33"

\header {
  title = "163 - Für Freuden laßt uns springen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 d4 d4 c4 
    bes2 a4 d4 
    d8 ees8 f4 ees4 d4 
    c2 d4 } d4 
  c4 bes4 a4 g4   %5
  a4 bes4 a4 d4 
  bes4 bes4 c4 d4 
  ees4 d4 c4 c4 
  d4. c8 b4
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g8  fis8 
    g4 fis8 g8 a8 g4 fis8 
    g8 fis8 g4 fis4 f4 
    g4 f4 c'4. bes8 
    bes4 a4 bes4 } bes8 a8 
  g4 g4. fis8 g4   %5
  c,8 a'8 d,8 e8 fis4 a4~ 
  a4 g8 f4 ees8 aes8 g8 
  g4 g4 a4 a8 g8 
  fis8 g4 fis8 g4
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes8  c8 
    d8 c16 bes16 a4. bes8 c8 d8 
    d4 d4 d4 d8 c8 
    bes4 bes4. a16 g16 f8 f'8 
    g4 f4 f4 } f4 
  ees8 d8 e4 a,8 d16 c16 bes8 c16 bes16   %5
  a8 d16 c16 bes16 a16 g8 d'4 d4 
  d4 ees8 f8 g4 f8 d8~ 
  d8 c8 d8 e8 fis8 a,8 d8 ees8 
  d8 c16 bes16 a8 d8 d4
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g8  a8 
    bes8 c8 d8 e8 fis8 g8 a8 d,8 
    g8 a8 bes8 c8 d4 bes8 a8 
    g8 f8 ees8 d8 c8 f8 bes,8 d8 
    ees8 c8 f4 bes,4 } bes4 
  c4. cis8 d4 ees8 e8   %5
  f8 fis8 g8 cis,8 d4 fis4 
  g8 f8 ees8 d8 c4. b8 
  c8 c'4 bes8 a8 g8 fis8 g8 
  d2 g4
  
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